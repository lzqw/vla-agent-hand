# 纯关节动作版 RaboVLA

现在两套策略对网页端都只返回：

```json
{
  "type": "action",
  "action_space": "joint_position_36d",
  "action": [36个float]
}
```

36维顺序固定：`left_arm(7) + right_arm(7) + left_hand(11) + right_hand(11)`。

## A. 无训练：joint_vla

不在4080伪造A7 URDF/DH。使用官方Rabo SDK在成功Expert轨迹中实际求解/执行后记录的36D关节轨迹，当前 `full_state` 与参考轨迹前向匹配，然后输出下一帧关节目标。

准备单条LeRobot episode bundle后：

```bash
cd ~/vla_bridge
. .venv/bin/activate
pip install -r requirements.txt
python -m tools.prepare_joint_dataset \
  ~/vla_bridge/data/bc/rabo_joint_bc_episode_000001 \
  --episode-index 000001
```

得到：

```text
~/vla_bridge/data/joint/reference_v1.npz
~/vla_bridge/data/joint/bc_episode_v1.npz
```

切换：

```bash
mkdir -p ~/.config/vla-bridge
cat > ~/.config/vla-bridge/runtime.env <<'EOF'
VLA_POLICY=joint_vla
JOINT_REFERENCE_PATH=/home/carla/vla_bridge/data/joint/reference_v1.npz
EOF
systemctl --user restart vla-bridge.service
curl http://127.0.0.1:8765/healthz
```

health 应显示 `policy=joint_vla`、`action_space=joint_position_36d`、`output_action_dim=36`。

## B. 训练：bc_joint_vla

训练的是：

```text
3 RGB + current full_state[36] -> next full_state[36]
```

请求中的 `step` 不进入神经网络。

CPU训练示例：

```bash
cd ~/vla_bridge
. .venv/bin/activate
python -m app.policies.bc_joint.train \
  --cache ~/vla_bridge/data/joint/bc_episode_v1.npz \
  --output-dir ~/vla_bridge/models/rabo_bc_joint_v1 \
  --device cpu \
  --epochs 60 \
  --samples-per-epoch 2048 \
  --batch-size 32
```

训练完成后：

```bash
cat > ~/.config/vla-bridge/runtime.env <<'EOF'
VLA_POLICY=bc_joint_vla
BC_JOINT_MODEL_DIR=/home/carla/vla_bridge/models/rabo_bc_joint_v1
EOF
systemctl --user restart vla-bridge.service
curl http://127.0.0.1:8765/healthz
```

health 应显示 `policy=bc_joint_vla`、`model=RaboBC-Joint-v1`、`action_space=joint_position_36d`。

## 网页端

新版网页端默认：

```text
VLA_EXECUTE_ACTIONS=1
VLA_EXECUTE_REMOTE_COMMANDS=0
VLA_CONTROL_HZ=5
```

收到36D action后按四个设备切片，通过官方 `move_joints(..., blocking=False)` 执行，并在本地做 joint-limit 与每周期 slew-rate 安全限制。

旧 `rabo_vla` structured action 仍留作回退，但两套新策略都不会输出 mixed command。
