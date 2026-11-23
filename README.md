# starvla_rebot

ReBot 双臂策略。Qwen2.5-VL-3B 读取三路相机图像、双臂状态和任务指令，OFT 动作头一次预测未来 50 帧。

## 训练与部署

在仓库根目录运行。示例使用 `rebot_data/packing` 下的 30 Hz LeRobot v3.0 数据，重复训练命令会从已有 checkpoint 续训。

```bash
conda activate starvla_rebot
starvla-rebot train --data rebot_data/packing --output rebot_runs/packing
starvla-rebot serve --checkpoint rebot_runs/packing/last.pt
```

服务启动后，在控制端运行 `rebot policy --port 8011`。

默认基座路径为 `rebot_models/Qwen2.5-VL-3B-Instruct`。复制 checkpoint 时，将同目录的 `processor/` 一并带上。

## 新机器安装

将 LOOM-policy 放在相邻目录，并把下面的 `/path/to/rebot_control` 换成控制器源码路径。

```bash
conda env create -f environment-rebot.yml
conda activate starvla_rebot
python -m pip install -e ../LOOM-policy /path/to/rebot_control/policy_link -e '.[rebot]'
```

[启智集群路径](INSPIRE.md)
