"""项目入口: 选择并启动一个 agent。

Rabo 平台默认以 ``python3 -u main.py`` 无参启动。默认运行远程 VLA。
需要采 expert 数据时，可在工程根目录写 `.agent_mode`：

    echo expert_collect > .agent_mode

采完恢复远程 VLA：

    echo vla_agent > .agent_mode

优先级：命令行参数 > RABO_AGENT 环境变量 > .agent_mode > DEFAULT_AGENT。
"""

import importlib
import os
import sys
from pathlib import Path


DEFAULT_AGENT = "vla_agent"
MODE_FILE = Path(__file__).resolve().with_name(".agent_mode")


def _selected_agent() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    env_name = os.getenv("RABO_AGENT", "").strip()
    if env_name:
        return env_name
    try:
        file_name = MODE_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        file_name = ""
    return file_name or DEFAULT_AGENT


def main():
    name = _selected_agent()
    print(f"[main] agent={name}", flush=True)
    try:
        module = importlib.import_module(f"agents.{name}")
    except ModuleNotFoundError as e:
        print(f"找不到 agent: {name} ({e})", file=sys.stderr)
        print("可用 agent 在 agents/ 目录下", file=sys.stderr)
        sys.exit(1)

    if not hasattr(module, "run"):
        print(f"agents/{name}/__init__.py 必须暴露 run() 函数", file=sys.stderr)
        sys.exit(1)

    module.run()


if __name__ == "__main__":
    main()
