"""ExampleAgent: 演示框架用法的最小 agent (无机器人, 无 ROS)。

────────────────────────────────────────────────────────────────────────
你可以把这个文件作为新 agent 的起点。要做的事情很少:

  1. 在 __init__ 里初始化外部依赖 (硬件 SDK / 传感器订阅 / 数据库连接 …)
  2. 在 __init__ 里设好 BaseAgent 协议要求的 4 个属性:
        self.llm, self.model, self.system_prompt, self.tools
     然后调 self.setup_messages()
  3. 实现 execute_tool(name, args) -> str
        一般直接转给 .tools 模块里的 execute_tool

如果你的 agent 不需要 ROS 或长期后台进程, 这一个文件就完整定义了你的 agent。
如果需要 ROS, 让你的 agent 类同时继承 BaseAgent 和 rclpy.Node:

    class MyAgent(BaseAgent, Node):
        def __init__(self):
            Node.__init__(self, "my_agent")
            BaseAgent.__init__(self)
            ...

在 rabo 平台上接机器人本体的写法见 README「教程四」(用 rabo_robocap)。
────────────────────────────────────────────────────────────────────────
"""

# 接入大模型时启用(与下方 LLM 接入代码一起取消注释):
# from openai import OpenAI

from core import BaseAgent

from . import config
from . import tools as tools_module
# 接入大模型时启用(下方 LLM 接入代码会用到):
# from .prompts import SYSTEM_PROMPT
# from .tools import TOOLS


class ExampleAgent(BaseAgent):
    """最小示例 agent: 演示工具调用框架, 不依赖任何机器人或 ROS。"""

    def __init__(self):
        super().__init__()

        # ── 1. agent 自己的状态 ───────────────────────────────
        # 任何跨工具调用要共享的数据都放在 self 上。这里只是一个笔记列表。
        # 真实场景里这里可能是: self.arm = ..., self.camera = ..., 等。
        self.notes: list[str] = []

        # ── 2. 接入大模型(设置 BaseAgent 协议要求的 4 个属性:
        #       self.llm 客户端 / self.model 模型名 / self.system_prompt 系统提示 / self.tools 工具表)──
        # 模板默认保持最小可运行,不接 LLM。接入时:先在 config.py 填好 LLM 配置(并用
        # RABO_LLM_KEY 提供 key),再取消下面整段、以及顶部 OpenAI / SYSTEM_PROMPT / TOOLS
        # 的 import 注释,agent 即可调用大模型与工具。
        # self.llm = OpenAI(
        #     base_url=config.LLM_BASE_URL,
        #     api_key=config.LLM_API_KEY,
        # )
        # self.model = config.LLM_MODEL
        # self.system_prompt = SYSTEM_PROMPT
        # self.tools = TOOLS
        # self.setup_messages()  # 用 system_prompt 初始化对话历史

        self.logger.info("ExampleAgent 初始化完成。")

    def execute_tool(self, name: str, args: dict) -> str:
        """BaseAgent 钩子: 把工具调度转给 tools 模块。"""
        return tools_module.execute_tool(self, name, args)
