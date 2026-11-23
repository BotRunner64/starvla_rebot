"""Qwen2.5-VL and StarVLA's parallel continuous OFT head for ReBot."""
import os
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel

from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead


def add_options(parser):
    local = Path(__file__).resolve().parents[1] / 'rebot_models' / 'Qwen2.5-VL-3B-Instruct'
    default = str(local) if (local / 'model.safetensors.index.json').is_file() else 'Qwen/Qwen2.5-VL-3B-Instruct'
    parser.add_argument('--backbone', default=default,
                        help='Qwen2.5-VL pretrained directory or Hugging Face ID')


def model_options(args):
    return {'backbone': args.backbone}


def discretize(values):
    bins = ((values.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().int()
    return ' '.join(map(str, bins.flatten().tolist()))


class Model(nn.Module):
    def __init__(self, horizon, options, directory, new):
        super().__init__()
        self.horizon = horizon
        assets = directory / 'processor'
        if new:
            self.processor = AutoProcessor.from_pretrained(options['backbone'], use_fast=False)
            pretrained = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                options['backbone'], torch_dtype=torch.float32, attn_implementation='sdpa')
            self.vlm = pretrained.model
            if int(os.environ.get('RANK', 0)) == 0:
                self.processor.save_pretrained(assets)
                self.vlm.config.save_pretrained(assets)
        else:
            self.processor = AutoProcessor.from_pretrained(assets, use_fast=False)
            config = Qwen2_5_VLConfig.from_pretrained(assets)
            config._attn_implementation = 'sdpa'
            self.vlm = Qwen2_5_VLModel(config)
        self.processor.tokenizer.padding_side = 'left'
        # Append an existing vocabulary ID explicitly, one parallel slot per future step.
        self.action_id = self.processor.tokenizer.encode('🔍', add_special_tokens=False)[0]
        hidden = self.vlm.config.text_config.hidden_size
        self.head = L1RegressionActionHead(hidden, 2 * hidden, 20, horizon)

    def predict(self, rgb, state, text, prefix, mask):
        images = rgb.detach().cpu().permute(0, 1, 3, 4, 2).numpy()
        prompts, views = [], []
        for frame, instruction, pose, known, valid in zip(images, text, state, prefix, mask, strict=True):
            content = [{'type': 'image'} for _ in frame]
            prompt = f'{instruction}\nState: {discretize(pose)}'
            if valid.any():
                prompt += f'\nCommitted actions: {discretize(known[valid])}'
            content.append({'type': 'text', 'text': f'{prompt}\nPredict the next {self.horizon} actions.'})
            prompts.append(self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                               tokenize=False, add_generation_prompt=True))
            views.extend(Image.fromarray(view) for view in frame)
        inputs = self.processor(text=prompts, images=views, return_tensors='pt', padding=True).to(state.device)
        slots = inputs.input_ids.new_full((len(state), self.horizon), self.action_id)
        inputs['input_ids'] = torch.cat((inputs.input_ids, slots), dim=1)
        inputs['attention_mask'] = torch.cat((inputs.attention_mask, torch.ones_like(slots)), dim=1)
        hidden = self.vlm(**inputs, use_cache=False).last_hidden_state
        return self.head(hidden[:, -self.horizon:])

    def loss(self, rgb, state, text, target, prefix, mask):
        prediction = self.predict(rgb, state, text, prefix, mask)
        return F.l1_loss(prediction[~mask].float(), target[~mask])
