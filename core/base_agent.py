"""Agent 框架核心: LLM 工具调用循环 + 消息史管理。

这是整个模板的"心脏": 它负责跟 LLM 来回对话、把 LLM 想用的工具调起来、把
工具结果回灌给 LLM, 直到 LLM 输出最终文本回复。具体 agent (机械臂、巡检、
对话客服…) 在子类里只需要补三件事: 工具列表、prompt、工具的实际执行逻辑。

本文件不导入任何具体业务依赖 (机器人 SDK / ROS / 视觉模型), 这样就可以被
任意 agent 复用。

────────────────────────────────────────────────────────────────────────
读这个文件的顺序建议:
  1. 先看 BaseAgent.run() —— 整个循环的主流程
  2. 再看子类要填的钩子 (execute_tool, system_prompt, tools, llm, model)
  3. 最后看 safe_parse_args —— 处理 LLM 偶发输出非法 JSON 的兜底
────────────────────────────────────────────────────────────────────────
"""

import ast
import json
import logging
import operator
import threading


# ══════════════════════════════════════════════════════════════════════
#  工具参数容错解析
# ══════════════════════════════════════════════════════════════════════
#
# Qwen / DeepSeek / Kimi 等模型偶发会输出非法 JSON 作为工具参数, 比如:
#
#     {"x": 00.628}            ← 前导零, json.loads 会报错
#     {"z": -0.1 + 0.15}       ← 算术表达式
#     {'name': True}           ← Python 字面量而不是 JSON
#
# 这些非法字符串如果直接 append 进 messages, 下一轮请求会被服务端 400 拒绝,
# 整个 session 就废了。所以我们用 AST 安全求值兜底, 把它修正成合法 JSON 再
# 写回 tool_call.arguments。
#
# 参考: https://github.com/MoonshotAI/kimi-cli/issues/1171

_AST_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_ast(node):
    """递归对一个被 ast.parse 解析出的简单表达式求值。

    只支持常量、一元/二元算术、dict/list/tuple 字面量。
    其它任何节点 (函数调用、属性访问、import 等) 都会抛 ValueError。
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        return _AST_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp):
        return _AST_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.Dict):
        return {_eval_ast(k): _eval_ast(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [_eval_ast(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(e) for e in node.elts)
    raise ValueError(f"unsupported AST node: {type(node).__name__}")


def safe_parse_args(s: str) -> dict:
    """把一段可能不合法的 JSON 字符串安全解析成 dict。

    优先走 json.loads (快路径)。只有失败时才走 AST 兜底, 把 Python 字面量
    与算术表达式也当合法输入。返回值必定是 dict, 否则抛 ValueError。
    """
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    py_s = s.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    tree = ast.parse(py_s, mode='eval')
    result = _eval_ast(tree.body)
    if not isinstance(result, dict):
        raise ValueError(f"expected dict, got {type(result).__name__}")
    return result


# ══════════════════════════════════════════════════════════════════════
#  BaseAgent
# ══════════════════════════════════════════════════════════════════════

class BaseAgent:
    """LLM agent 基类: 负责"用户文本 → 多轮工具调用 → 最终回复"的整套循环。

    ────────────────────────────────────────────────────────────────────
    子类协议 (你写新 agent 时必须做的事)

    在 __init__ 里设置以下 4 个属性, 然后调 self.setup_messages():
        self.llm           OpenAI 兼容客户端 (openai.OpenAI 实例)
        self.model         LLM 模型名 (字符串)
        self.system_prompt 系统提示词 (字符串)
        self.tools         OpenAI function schema 列表

    实现以下方法:
        execute_tool(name, args) -> str
            根据工具名和参数实际执行操作, 返回字符串结果。
            一般做法: 转给 agents/<name>/tools.py 里的调度函数。

    然后外部代码调用 agent.run(user_text) 就能得到最终回复字符串。
    ────────────────────────────────────────────────────────────────────

    可调参数 (子类可覆盖):
        MAX_ROUNDS    单次 run() 最多容许多少轮工具调用 (防死循环)
        TEMPERATURE   LLM 采样温度。工具调用类任务建议 0.0 (稳定)
    """

    MAX_ROUNDS = 20
    TEMPERATURE = 0.0

    def __init__(self):
        # 完整的对话历史。setup_messages() 会注入 system prompt。
        # 每次 run() 会把用户消息、LLM 回复、工具结果都 append 进来。
        self.messages = []

        # 锁: 防止 run() 被并发调用 (例如多个 ROS 回调同时触发)。
        # 业务代码若用线程跑 run(), 用这个锁就能保证消息史一致。
        self._lock = threading.Lock()

        # 默认 logger。如果 agent 跑在 ROS 节点里, 子类可以覆盖成 ROS logger。
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_messages(self):
        """以 self.system_prompt 初始化对话历史。子类在 __init__ 末尾调一次。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def execute_tool(self, name: str, args: dict) -> str:
        """子类必须实现: 根据工具名执行操作, 返回字符串结果。

        约定: 任何错误都应该被 catch 并返回错误描述字符串, 而不是抛异常。
        让 LLM 看到错误描述, 它就有机会重试或反问用户。
        """
        raise NotImplementedError("子类需实现 execute_tool")

    def run(self, user_text: str) -> str:
        """主循环: 发用户消息 → LLM 决定调工具 → 执行 → 回灌 → … → 文本回复。

        循环退出条件:
          (a) LLM 返回的消息没有 tool_calls (说它要直接回答用户) → 返回它的文本
          (b) 达到 MAX_ROUNDS 轮还没收敛 → 返回中止提示
        """
        with self._lock:
            # 1. 把用户消息加进对话历史
            self.messages.append({"role": "user", "content": user_text})
            self.logger.info(f"[agent] 开始处理, 当前对话历史 {len(self.messages)} 条")

            # 2. 多轮循环: 每一轮 = 一次 LLM 请求 + 0 或多次工具调用
            for round_i in range(self.MAX_ROUNDS):
                self.logger.info(f"[agent] 第 {round_i + 1} 轮, 调用 {self.model}...")

                # 2a. 请求 LLM
                resp = self.llm.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    temperature=self.TEMPERATURE,
                    parallel_tool_calls=False,  # 串行更可控, 调试也容易
                )
                msg = resp.choices[0].message

                # 2b. 兜底修复 LLM 偶发输出的非法 JSON 工具参数,
                #     避免 session history 被毒化导致下一轮请求被服务端 400。
                for tc in (msg.tool_calls or []):
                    raw = tc.function.arguments or ""
                    try:
                        json.loads(raw) if raw else None
                    except json.JSONDecodeError:
                        try:
                            parsed = safe_parse_args(raw)
                            tc.function.arguments = json.dumps(parsed)
                            self.logger.warning(
                                f"[agent] tool_call 参数已修复: {raw!r} -> {tc.function.arguments}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"[agent] tool_call 参数无法解析, 置为空: {raw!r} ({e})"
                            )
                            tc.function.arguments = "{}"

                # 2c. 把 LLM 这一轮的回复 (无论文本还是工具调用) 加入历史。
                #     注意: 这里 append 的是 SDK 对象 (ChatCompletionMessage),
                #     而用户/工具消息是普通 dict —— self.messages 是混合类型。
                #     若要持久化对话历史, 不能直接 json.dumps; 见 README「常见问题」
                #     里的持久化说明 (用 msg.model_dump() 转 dict 最省事)。
                self.messages.append(msg)

                # 2d. 没有工具调用 → LLM 想直接回复用户, 结束循环
                if not msg.tool_calls:
                    self.logger.info(f"[agent] 第 {round_i + 1} 轮完成, 返回文本回复")
                    return msg.content or ""

                # 2e. 有工具调用 → 一个一个执行, 把结果以 role=tool 写回历史
                self.logger.info(
                    f"[agent] 第 {round_i + 1} 轮, {len(msg.tool_calls)} 个工具调用"
                )
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        self.logger.info(f"[agent] 调用工具: {tc.function.name}({args})")
                        result = self.execute_tool(tc.function.name, args)
                    except Exception as e:
                        # 安全网: 即使 execute_tool 没正确捕获异常, 这里也兜底,
                        # 把错误信息回灌给 LLM 让它有机会处理。
                        result = f"工具调用出错: {e}"
                    self.logger.info(f"[agent] 工具结果: {result[:200]}")
                    # 注意: tool_call_id 必须与 LLM 给出的 id 严格对应,
                    # 否则有些服务端 (如 OpenAI) 会拒绝下一轮请求。
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )

            # 3. 超过 MAX_ROUNDS 还没收敛, 一般是 LLM 陷入工具循环。中止并提示。
            self.logger.warning(f"[agent] 达到最大轮次 {self.MAX_ROUNDS}, 中止")
            return "操作步骤过多, 已中止。"
