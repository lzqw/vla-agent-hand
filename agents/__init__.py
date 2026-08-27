"""所有具体 agent 的容器目录。

每个子目录就是一个 agent (例如 example_agent/, arm_agent/, inspect_agent/),
通过 `python main.py <子目录名>` 启动。

加新 agent 的最简方式: 复制 example_agent/ 改名, 然后改里面的 4 件东西
(config / prompts / tools / agent)。详见 ../README.md。
"""
