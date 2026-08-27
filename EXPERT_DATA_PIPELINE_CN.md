# Expert command 数据采集与 4080 使用

## 为什么重新采一条

上传的旧 `rabo_nut_handoff_simple_v1_3cam_v3` 有 26 条固定场景成功轨迹，其中 `episode_000001` 质量最好，可留作后续 26D 连续动作 BC/VLA 的参考。但旧数据的 `action` 是 `state[t+1]`，不是当前远程协议使用的 structured `ActionCommand`。

因此当前 4080 Fixed-Oracle 测试优先重新采一条 command-level expert episode。

## 网页端采集

默认 `main.py` 仍运行 `vla_agent`。采 Expert 时设置：

```bash
export RABO_AGENT=expert_collect
export EXPERT_EPISODES=1
```

然后启动仿真并“运行智能体工程”。

数据默认写到：

```text
expert_data/episode_000000/
├── meta.json
├── steps.jsonl
├── expert_program.json
└── images/
    ├── cam_high/
    ├── cam_left_wrist/
    └── cam_right_wrist/
```

`steps.jsonl` 每条包含：三相机、26D state、36D full_state、phase、previous_command、command、执行结果。

`expert_program.json` 只包含 4080 lookup 所需的 59 条 `rabo_command_v1` command，是当前最重要的导出文件。

采完后人工确认 B/C/A 都完成，再把 `expert_program.json` 拷到 4080，例如：

```bash
scp expert_data/episode_000000/expert_program.json 4080:~/vla_bridge/data/expert_program.json
```

## 4080 端目标

Fixed Oracle 不再硬编码轨迹，而是启动时读取 `expert_program.json`：

```text
收到 state(step=N)
→ commands[N].command
→ 原样返回给网页端
```

网页端继续负责官方 SDK 的 `move_to / clench / grasp_force` 执行。

真正 VLA 接入以后，只替换 4080 的 lookup policy；网页端 observation/command 协议无需改。
