# VLA Bridge

当前 systemd 服务以 `VLA_POLICY=expert_lookup` 运行，用于打通 Rabo 仿真网页与 4080
之间的结构化动作闭环。它不会加载或训练 VLA，也不会在 4080 上计算 A7 IK；服务只返回
Cartesian `move_to` 与手部命令，实际 IK/执行由网页端官方 Rabo SDK 完成。

Expert Lookup 在启动时读取 `data/expert_program.json`。该文件来自仓库上层
`experts/fixed_scene_v1.json`，使用已验证的 B → C → A 参数，共 59 步；每次直接用请求中的
`step` 查表，不在服务端执行 IK 或维护隐藏动作游标。

## 协议

WebSocket 地址为 `/v1/ws`。继续支持 bearer header、`?token=` query 和 hello-token
三种认证方式；hello-token 示例：

```json
{"type":"hello","protocol":"vla-bridge.v1","token":"<TOKEN>","client":"simulation-web"}
```

认证成功后，每一步发送 `rabo_command_v1` state：

```json
{
  "type": "state",
  "protocol": "rabo_command_v1",
  "request_id": "episode-1-step-0",
  "episode_id": "episode-1",
  "step": 0,
  "instruction": "move B through C to A",
  "state": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "full_state": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "images": {
    "cam_high": {"encoding": "jpeg_base64", "data": "..."},
    "cam_left_wrist": {"encoding": "jpeg_base64", "data": "..."},
    "cam_right_wrist": {"encoding": "jpeg_base64", "data": "..."}
  }
}
```

`state` 必须为 26 维，`full_state` 必须为 36 维，instruction 非空，且三路相机必须全部
存在。服务直接返回完整 structured action，例如：

```json
{
  "type": "action",
  "protocol": "rabo_command_v1",
  "request_id": "episode-1-step-0",
  "episode_id": "episode-1",
  "oracle_step": 0,
  "phase": "approach_B",
  "command": {
    "action_type": "right_arm_move_to",
    "right_moves": [{"label": "右臂接近B", "pose": [-0.2803, 0.157, -0.331, 0, 0.8, 0]}]
  },
  "backend": "expert_lookup"
}
```

HTTP fallback 为 `POST /v1/action`，请求体与 WebSocket state 完全一致，并继续使用
`Authorization: Bearer <TOKEN>`。token 文件仍为 `~/.config/vla-bridge/token`。

健康检查至少包含：

```json
{"status":"ok","policy":"expert_lookup","expert_loaded":true,"expert_steps":59}
```

## 4080 运维

先准备 Expert 文件：

```bash
mkdir -p ~/vla_bridge/data
cp ~/vla-agent-hand/experts/fixed_scene_v1.json ~/vla_bridge/data/expert_program.json
```

服务固定使用现有端口 8765，uvicorn 监听 `0.0.0.0`；Quick Tunnel 仍转发到
`127.0.0.1:8765`。

```bash
systemctl --user status vla-bridge.service vla-bridge-tunnel.service --no-pager
journalctl --user -u vla-bridge.service -n 100 --no-pager
curl http://127.0.0.1:8765/healthz
```
