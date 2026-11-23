# 启智共享资产

在可联网的 CPU 实例准备环境，GPU 节点通过共享文件系统使用同一份代码和依赖。以下路径中的 `WORKSPACE_ROOT` 为用户共享工作目录，`PROJECT_PUBLIC` 为项目大容量共享目录。

| 资产 | 路径 |
|---|---|
| 仓库 | `$WORKSPACE_ROOT/projects/my_projects/starvla_rebot` |
| Conda 环境 | `$WORKSPACE_ROOT/cache/conda/envs/starvla_rebot` |
| `rebot_data` | `$PROJECT_PUBLIC/dataset/rebot` |
| `rebot_runs` | `$PROJECT_PUBLIC/model/starvla_rebot/runs` |
| 通信包 wheel | `$WORKSPACE_ROOT/cache/rebot_policy_link-0.1.0-py3-none-any.whl` |
| Qwen2.5-VL 基座 | `$PROJECT_PUBLIC/model/Qwen2.5-VL-3B-Instruct`（`rebot_models/Qwen2.5-VL-3B-Instruct`） |

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate $WORKSPACE_ROOT/cache/conda/envs/starvla_rebot
cd $WORKSPACE_ROOT/projects/my_projects/starvla_rebot
```

Packing 的同批 P1/P99 统计文件位于 `../openpi_rebot/data/packing_rtc_gop2_20261119_h50/rebot/norm_stats.json`。训练命令和服务参数见 README。
