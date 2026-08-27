# RaboBC-VLA-v1

这是固定场景、单成功 episode 的监督式 Behavior Cloning baseline。模型学习：

```text
3 x RGB (224x224) + state[26] -> logits[num_actions] -> action_id
```

网络没有 `step` 输入；`steps.jsonl` 中的 step 只作为监督 label 和离线评估 GT。language 和
36D full state 会由 policy 接收并校验，v1 暂不编码。模型默认也不使用 previous action；训练
时可显式加 `--use-previous-action`，推理值只能来自 server-side 上一次模型输出。

## 数据

默认 episode：

```text
~/vla_bridge/data/bc/episode_000000/
├── meta.json
├── steps.jsonl
├── expert_program.json
└── images/{cam_high,cam_left_wrist,cam_right_wrist}/
```

加载器会拒绝失败记录、非连续 action label、非 26D/36D state、不可读取的三路图片，以及与
`expert_program.json` 不一致的 command。

## 训练

4080 已有 `behavior` 环境包含兼容当前驱动的 CUDA PyTorch。禁止让用户目录中较新的
CUDA 13 PyTorch 覆盖它，因此训练时必须设置 `PYTHONNOUSERSITE=1`：

```bash
cd ~/vla_bridge
PYTHONNOUSERSITE=1 \
~/anaconda3/envs/behavior/bin/python -m app.policies.bc.train \
  --episode ~/vla_bridge/data/bc/episode_000000 \
  --output ~/vla_bridge/models/rabo_bc_v1 \
  --epochs 200 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4
```

每个 epoch 默认从原始记录动态采样 100 次；augmentation 只包含 0.92--1.0 轻 crop、
小于 2% translation、轻 color jitter、微小像素噪声与 0.003 proprio noise。没有 flip 或旋转。

训练输出：

```text
~/vla_bridge/models/rabo_bc_v1/
├── model.pt
├── config.json
├── metrics.json
└── action_library.json
```

`metrics.json` 包含 stochastic train/validation accuracy、original episode exact/phase
accuracy、confusion matrices、逐 step prediction、confidence 和 sequence completion。

## Shadow

生产 unit 保持 `VLA_POLICY=rabo_vla`。只有人工复制 shadow drop-in 后才加载 BC：

```bash
mkdir -p ~/.config/systemd/user/vla-bridge.service.d
cp ~/vla-agent-hand/vla_bridge/service/vla-bridge-bc-shadow.conf \
  ~/.config/systemd/user/vla-bridge.service.d/90-bc-shadow.conf
systemctl --user daemon-reload
systemctl --user restart vla-bridge.service
curl http://127.0.0.1:8765/healthz
```

shadow 模式仍返回 Expert/RaboVLA action 给网页执行，只记录 BC prediction。恢复生产：

```bash
rm ~/.config/systemd/user/vla-bridge.service.d/90-bc-shadow.conf
systemctl --user daemon-reload
systemctl --user restart vla-bridge.service
```
