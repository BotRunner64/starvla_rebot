"""Train or serve a ReBot action-chunk policy."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from loom_policy.config import Config
from loom_policy.data import CAMERAS, Samples, find_datasets, resize, statistics
from loom_policy.geometry import compose, relative
from loom_policy.serve_rebot import RebotPolicy
import rebot_policy_link as link

from .model import Model, add_options, model_options


def collate(samples):
    """Keep three synchronized RGB views in controller order."""
    segments = [sample['segments'][0] for sample in samples]
    if any(segment['cameras'] != list(range(len(CAMERAS))) for segment in segments):
        raise ValueError(f'Training requires all three cameras: {CAMERAS}')
    rgb = torch.stack([torch.stack(seg['images'][0]) for seg in segments])
    state = torch.stack([seg['state'][0] for seg in segments])
    text = [bytes(seg['texts'][0][1].tolist()).decode() for seg in segments]
    return rgb, state, text, torch.stack([sample['target'] for sample in samples])


class Policy(nn.Module):
    def __init__(self, config, stats, options, directory, new):
        super().__init__()
        self.config = config
        self.network = Model(config.horizon, options, directory, new)
        for key, values in stats.items():
            self.register_buffer(key, torch.tensor(values, dtype=torch.float32))

    def forward(self, rgb, state, text, target):
        target = (target - self.action_center) / self.action_scale
        lengths = torch.randint(self.config.max_prefix_len + 1, (len(target),), device=target.device)
        mask = torch.arange(self.config.horizon, device=target.device)[None, :] < lengths[:, None]
        prefix = torch.where(mask[..., None], target, 0.)
        return self.network.loss(rgb, (state - self.input_center) / self.input_scale, text, target, prefix, mask)

    @torch.inference_mode()
    def act(self, states, images, cameras, instruction, prefix_states=None):
        if tuple(cameras) != CAMERAS:
            raise ValueError(f'Expected cameras in order {CAMERAS}')
        device = self.input_center.device
        base = np.asarray(states[-1], dtype=np.float32)
        views = np.stack([resize(view, self.config.image_size) for view in images[-1]])
        rgb = torch.as_tensor(views).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        state = (torch.as_tensor(base, device=device) - self.input_center) / self.input_scale
        prefix = torch.zeros(1, self.config.horizon, 20, device=device)
        count = 0 if prefix_states is None else len(prefix_states)
        if count > self.config.max_prefix_len:
            raise ValueError('Prefix exceeds training range')
        if count:
            encoded = torch.as_tensor(relative(base, prefix_states), device=device)
            prefix[0, :count] = (encoded - self.action_center) / self.action_scale
        mask = torch.arange(self.config.horizon, device=device)[None] < count
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            actions = self.network.predict(rgb, state[None], [instruction], prefix, mask)
        decoded = (actions[0].float() * self.action_scale + self.action_center).cpu().numpy()
        chunk = compose(base, decoded)
        if count:
            chunk[:count] = prefix_states
        return chunk


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def train(args):
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group('nccl' if cuda else 'gloo')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'last.pt'
    saved = load(path) if path.exists() else None
    roots = find_datasets(args.data)
    if saved:
        config = Config(**saved['config'])
        stats, options, step = saved['stats'], saved['options'], saved['step']
    else:
        config = Config(current_frames=1, absolute_state=True, max_refs=0, adaptive_rgb=False,
                        image_size=(168, 224), patch=14, val_fraction=0., max_prefix_len=6,
                        horizon=50, rebot_quantile_stats=args.stats)
        stats = statistics(roots, config) if rank == 0 else None
        if world > 1:
            payload = [stats]
            torch.distributed.broadcast_object_list(payload, src=0)
            stats = payload[0]
        options, step = model_options(args), 0
    torch.manual_seed(42)
    policy = Policy(config, stats, options, args.output, new=saved is None).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    if saved:
        policy.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        del saved
    parallel = DistributedDataParallel(policy, device_ids=[device.index] if cuda else None) if world > 1 else policy
    torch.manual_seed(42 + world * step + rank)
    samples = Samples(roots, config, seed=[42, step, rank])
    if any(source.cameras != [0, 1, 2] for source in samples.sources):
        raise ValueError(f'Expected three-view datasets {CAMERAS}')
    batches = iter(torch.utils.data.DataLoader(samples, batch_size=args.batch_size, collate_fn=collate,
                                               num_workers=args.workers))
    while step < args.steps:
        rgb, state, text, target = next(batches)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            loss = parallel(rgb.to(device), state.to(device), text, target.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if rank == 0:
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(f'step={step} loss={loss.item():.6f}', flush=True)
            if step % args.save_every == 0 or step == args.steps:
                tmp = path.with_suffix('.tmp')
                torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict(), step=step,
                                config=asdict(config), stats=stats, options=options), tmp)
                os.replace(tmp, path)
    if world > 1:
        torch.distributed.destroy_process_group()


def load_policy(path):
    saved = load(path)
    policy = Policy(Config(**saved['config']), saved['stats'], saved['options'], path.parent, new=False)
    policy.load_state_dict(saved['model'])
    return policy.eval().to('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--data', nargs='+', required=True)
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--stats', help='openpi_rebot norm_stats.json; otherwise estimate on training windows')
    fit.add_argument('--steps', type=int, default=20000)
    fit.add_argument('--batch-size', type=int, default=2)
    fit.add_argument('--workers', type=int, default=4)
    fit.add_argument('--lr', type=float, default=2e-5)
    fit.add_argument('--save-every', type=int, default=500)
    add_options(fit)
    server = commands.add_parser('serve')
    server.add_argument('--checkpoint', required=True, type=Path)
    server.add_argument('--port', type=int, default=8011)
    server.add_argument('--host', default='127.0.0.1')
    server.add_argument('--rtc-prefix-steps', type=int, default=3)
    args = parser.parse_args()
    if args.command == 'train':
        if min(args.steps, args.batch_size, args.save_every) < 1 or args.workers < 0:
            parser.error('steps, batch-size and save-every must be positive; workers must be nonnegative')
        train(args)
    else:
        infer = RebotPolicy(load_policy(args.checkpoint), args.rtc_prefix_steps)
        infer.warmup()
        meta = link.metadata(rgb=CAMERAS, rtc_prefix_steps=args.rtc_prefix_steps, policy='starvla_rebot')
        link.serve(infer, meta, args.host, args.port, ready=lambda _: print(f'Ready on {args.host}:{args.port}', flush=True))


if __name__ == '__main__':
    main()
