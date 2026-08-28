# 14D 双臂关节动作 + Expert 手事件

`joint_vla` 和 `bc_joint_vla` 对网页端使用同一协议：

```json
{
  "type": "action",
  "action_space": "arm_joint_position_14d",
  "action": ["left_arm(7)", "right_arm(7)"],
  "hand_command": null,
  "done": false
}
```

O6 手不使用 `move_joints`。抓取、力控、等待和释放继续复用成功
command Expert 中的 `clench / grasp_force / wait` 命令。某个事件到达时，
`hand_command` 为原始 command；每个 episode 只触发一次，reset 会清空轨迹 cursor
和 fired-event 集合。

## 数据准备

需要两份成功数据：

- 850 帧、5 Hz 的连续 episode：提供双臂下一帧 14D 监督目标和三路 RGB。
- 59 步 command Expert：提供执行前 36D 状态和已验证的 O6 手动作时机。

```bash
python -m tools.prepare_joint_dataset \
  ~/vla_bridge/data/bc/rabo_joint_bc_episode_000001 \
  --episode-index 000001 \
  --expert-steps ~/vla_bridge/data/bc/expert_data/episode_000000/steps.jsonl
```

输出：

```text
~/vla_bridge/data/joint/arm_hand_reference_v1.npz
~/vla_bridge/data/joint/hand_events_v1.json
~/vla_bridge/data/joint/bc_episode_v1.npz
~/vla_bridge/data/joint/dataset_meta.json
```

工具用 59 步 observation 的前 14D arm state 对 850 帧轨迹做单调 DTW
最近邻对齐，并只把 hand/wait command 写入事件表。训练 target 定义为：

```text
target_arm_state[t] = observation.full_state[t+1, :14]
```

最后一帧 target 等于最后一帧 arm state。

## joint_vla（不训练）

```text
live full_state[:14]
  -> dense reference 前向窗口匹配
  -> next arm joints[14]
  + optional one-shot hand_command
```

它不读取 request.step，也不在 4080 实现 A7 IK。

```bash
cat > ~/.config/vla-bridge/runtime.env <<'EOF'
VLA_POLICY=joint_vla
JOINT_REFERENCE_PATH=/home/carla/vla_bridge/data/joint/arm_hand_reference_v1.npz
HAND_EVENTS_PATH=/home/carla/vla_bridge/data/joint/hand_events_v1.json
JOINT_INITIAL_SEARCH=250
JOINT_FORWARD_WINDOW=80
EOF
```

## bc_joint_vla（监督训练）

模型学习：

```text
3 RGB + current full_state[36] -> next arm joints[14]
```

request.step 不进入网络，O6 的 22D 手关节也不是回归目标。手事件仍由同一个
`hand_events_v1.json` 调度。

```bash
python -m app.policies.bc_joint.train \
  --cache ~/vla_bridge/data/joint/bc_episode_v1.npz \
  --output-dir ~/vla_bridge/models/rabo_bc_joint_v1 \
  --device cpu \
  --epochs 60 \
  --samples-per-epoch 2048 \
  --batch-size 32
```

## 首次网页验证

先同时关闭 numeric arm 和 remote hand 执行：

```bash
VLA_EXECUTE_ACTIONS=0 \
VLA_EXECUTE_REMOTE_COMMANDS=0 \
VLA_MAX_CYCLES=5 \
python main.py
```

确认 `response_kind=joint_action`、`action_type=arm_joint_position_14d`、action
长度为 14，且 `hand_command` 仅在映射事件处出现后，再分别开启两个执行开关。
