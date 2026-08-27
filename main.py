"""项目入口: 选择并启动一个 agent。

Rabo 平台默认以 ``python3 -u main.py`` 无参启动。默认运行远程 VLA
智能体；需要采一条本地 expert command 数据时，将环境变量
``RABO_AGENT=expert_collect`` 即可切换，而无需修改代码。
"""

import importlib
import os
import sys


DEFAULT_AGENT = "vla_agent"


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = os.getenv("RABO_AGENT", DEFAULT_AGENT).strip() or DEFAULT_AGENT

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
