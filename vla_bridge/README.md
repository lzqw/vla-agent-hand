# RaboVLA Bridge

当前 4080 端统一以 `VLA_POLICY=rabo_vla` 运行，对网页端暴露稳定的多模态
`observation -> action` 接口。网页端不需要关心动作维度，也不需要知道 4080 内部具体使用
learned VLA、IK 还是专家控制器。

第一版模型名为 `RaboVLA-Hybrid-v1`。它接收三路 RGB、language instruction、26D
proprioception 和 36D full proprioception，并通过：

```python
action = await policy.act(observation)
```

返回可执行 robot action。

当前 v1 的内部实现仍使用已经完整验证过的 `expert_program.json` 作为动作后端，因此保持
B -> C -> A 闭环的可靠性；这属于 VLA-compatible hybrid controller，而不是声称已经训练了
一个神经网络 VLA。后续可把内部 backend 替换为 server-side IK、26D joint policy 或真实
DexVLA，而不修改网页端协议。

## Observation

WebSocket 地址为 `/v1/ws`，HTTP fallback 为 `POST /v1/action`。认证继续支持 bearer
header、`?token=` query 与 hello-token。

每一步 observation：

```json
{
  "type": "state",
  "protocol": "rabo_command_v1",
  "request_id": "episode-1-step-0",
  "episode_id": "episode-1",
  "step": 0,
  "instruction": "双臂协作依次抓取B、C、A螺母，由右手递交左手并放入目标区",
  "state": [26],
  "full_state": [36],
  "images": {
    "cam_high": {"encoding": "jpeg_base64", "data": "..."},
    "cam_left_wrist": {"encoding": "jpeg_base64", "data": "..."},
    "cam_right_wrist": {"encoding": "jpeg_base64", "data": "..."}
  }
}
```

当前 hybrid backend 会完整校验三路视觉、instruction、26D/36D proprio 输入。v1 的实际
动作选择仍由已验证的 expert program 提供；以后换 backend 时 observation schema 不变。

## Action

当前返回 structured robot action，例如：

```json
{
  "type": "action",
  "protocol": "rabo_command_v1",
  "request_id": "episode-1-step-0",
  "episode_id": "episode-1",
  "policy": "rabo_vla",
  "model": "RaboVLA-Hybrid-v1",
  "backend": "rabo_vla",
  "policy_step": 0,
  "phase": "approach_B",
  "action_space": "structured_robot_action",
  "command": {
    "action_type": "right_arm_move_to",
    "right_moves": [
      {"label": "右臂接近B", "pose": [-0.2803, 0.157, -0.331, 0, 0.8, 0]}
    ]
  },
  "implementation": "expert_program_backend"
}
```

网页端 `RemoteCommandExecutor` 继续执行 `move_to / clench / grasp_force / wait`，A7 IK
仍由官方 Rabo SDK 完成。未来即使改成 26D numeric action，也只需要替换4080 policy backend。

## Health

`GET /healthz` 会暴露 RaboVLA 运行状态，例如：

```json
{
  "status": "ok",
  "policy": "rabo_vla",
  "model": "RaboVLA-Hybrid-v1",
  "model_family": "vision_language_action",
  "vision_inputs": 3,
  "proprio_dim": 26,
  "full_proprio_dim": 36,
  "language_input": true,
  "action_space": "structured_robot_action",
  "implementation": "expert_program_backend",
  "controller_loaded": true,
  "controller_steps": 59
}
```

## 4080 运维

Expert controller 文件仍位于：

```text
~/vla_bridge/data/expert_program.json
```

服务固定监听 8765，Cloudflare Quick Tunnel 转发 `127.0.0.1:8765`。

```bash
systemctl --user status vla-bridge.service vla-bridge-tunnel.service --no-pager
journalctl --user -u vla-bridge.service -f
curl http://127.0.0.1:8765/healthz
```

服务文件默认设置：

```text
VLA_POLICY=rabo_vla
```

后续替换内部动作生成器时，应保持 `act(observation) -> action` 这一层接口不变。

## Supervised BC baseline

可选的 `BCVLAPolicy` 实现真正经过监督训练的
`RGB + 26D proprioception -> action_id -> rabo_vla_action_v1`。它不读取请求 step，默认先以
`BC_SHADOW_ONLY=1` 与 Expert backend 对照，且不会自动替换生产 `rabo_vla`。数据格式、训练
命令、指标和 shadow 切换方式见
[`app/policies/bc/README.md`](app/policies/bc/README.md)。

## 14D arm-joint policies

`joint_vla` 与 `bc_joint_vla` 使用统一的
`arm_joint_position_14d + optional hand_command` 协议。双臂输出连续 14D
关节目标，O6 手继续执行成功 command Expert 中的 clench/grasp-force/wait
语义；两套策略都不读取 request.step，也不回归 O6 的 22D 手关节。
数据对齐、训练和 shadow 验证流程见 [JOINT_VLA_CN.md](JOINT_VLA_CN.md)。
