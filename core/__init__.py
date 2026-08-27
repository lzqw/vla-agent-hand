"""通用 agent 框架。

这一层不依赖任何具体的 agent / 机器人 / LLM 提供商, 任何 agent 都可以 import。
当前只暴露 BaseAgent (LLM 工具调用循环)。如果将来你写了多 agent 通用的辅助代码
(比如长期记忆、统一的日志格式), 也放这里。
"""

from .base_agent import BaseAgent, safe_parse_args

__all__ = ["BaseAgent", "safe_parse_args"]
