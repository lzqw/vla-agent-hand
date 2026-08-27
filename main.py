"""项目入口: 选择并启动一个 agent。

────────────────────────────────────────────────────────────────────────
在 rabo 平台上: 控制器由平台以 `python3 -u main.py`(**不带参数**)自动启动 ——
没有传参的入口, 所以实际运行的永远是下面 DEFAULT_AGENT 指定的那个 agent。
**要换运行哪个 agent, 改 DEFAULT_AGENT 的值。**

(可选)本地手动调试时, 可以用命令行参数显式指定某个 agent —— 平台用不到这一项:
    python main.py                  # 启动 DEFAULT_AGENT(默认 example_agent)
    python main.py <your_agent>     # 启动 agents/<your_agent>/(子包须在 __init__.py 暴露 run())
────────────────────────────────────────────────────────────────────────

为什么用 importlib 动态加载 (而不是 if/elif 硬编码)?
  - 加新 agent 时不用改 main.py, 只要在 agents/ 下加个目录就行
  - main.py 永远只有这几行, 不会随 agent 数量膨胀

如果你有需要在所有 agent 启动前都做的全局副作用 (设环境变量、关闭 SSL 验证、
注入 monkey patch 等), 写在 import 之前, 例如:

    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 必须最先!
"""

import importlib
import sys


# 平台无参启动时运行的 agent —— 换 agent 就改这里(本地调试也可用命令行参数覆盖)。
DEFAULT_AGENT = "example_agent"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AGENT

    try:
        module = importlib.import_module(f"agents.{name}")
    except ModuleNotFoundError as e:
        print(f"找不到 agent: {name} ({e})", file=sys.stderr)
        print("可用 agent 在 agents/ 目录下", file=sys.stderr)
        sys.exit(1)

    if not hasattr(module, "run"):
        print(
            f"agents/{name}/__init__.py 必须暴露 run() 函数",
            file=sys.stderr,
        )
        sys.exit(1)

    module.run()


if __name__ == "__main__":
    main()
