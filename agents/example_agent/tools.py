"""ExampleAgent 的工具集: schema + handler + 调度。

════════════════════════════════════════════════════════════════════════
本文件是 "agent 加新能力" 的唯一入口。给 agent 加一个工具的标准三步:

  ┌─ 步骤 1 ─────────────────────────────────────────────────┐
  │ 在 TOOLS 列表加一个 OpenAI function schema:              │
  │   - name: 工具名 (英文 snake_case, LLM 看到这个去识别)   │
  │   - description: 一句话说清"什么时候该调它"              │
  │   - parameters: JSON Schema, 描述参数                    │
  └──────────────────────────────────────────────────────────┘

  ┌─ 步骤 2 ─────────────────────────────────────────────────┐
  │ 写一个 _xxx(agent, args) -> str 函数:                    │
  │   - 第一个参数是 agent 实例, 用它访问硬件/状态/LLM       │
  │   - 第二个参数是 dict, 由 LLM 给的 JSON 参数             │
  │   - 返回字符串 (LLM 会把它当工具结果读)                  │
  │   - 错误信息用 return 返回, 不要 raise                   │
  └──────────────────────────────────────────────────────────┘

  ┌─ 步骤 3 ─────────────────────────────────────────────────┐
  │ 在 _HANDLERS 字典加一行:  "tool_name": _xxx              │
  └──────────────────────────────────────────────────────────┘

完事。重启 agent, LLM 下一轮就能用新工具了。
════════════════════════════════════════════════════════════════════════

handler 函数命名约定: 文件内私有函数加下划线前缀 (_my_tool), 这样能跟工具
名 (my_tool) 区分, 又避免污染 import * 的命名空间。
"""

import ast
import datetime
import operator


# ══════════════════════════════════════════════════════════════════════
#  工具 schema 列表 (OpenAI function calling 格式)
# ══════════════════════════════════════════════════════════════════════
#
# 每一项就是 LLM 拿到的一个 "可用工具描述"。LLM 决定调谁、传什么参数,
# 全靠它。所以 description 要写清楚, parameters 的 description 也要写清楚。

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前的本地日期与时间, 用于回答用户'现在几点/几号'",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "对一段算术表达式求值, 支持 + - * / 与括号。用于回答用户的算术问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的算术表达式, 例如 '2 + 2' 或 '(60*60*24)*365'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note",
            "description": "把一条文字记进 agent 的内部记事本, 后续可通过 list_notes 取回",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要记下的内容",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "列出所有已记下的笔记。如果一条都没有, 返回提示。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ══════════════════════════════════════════════════════════════════════
#  handler 实现
# ══════════════════════════════════════════════════════════════════════
#
# 每个 handler 的签名都是  _xxx(agent, args) -> str。
# - agent: ExampleAgent 实例。需要状态就用 agent.notes 这种属性。
#                  需要硬件就用 agent.arm / agent.camera (在你自己的 agent 里)。
# - args:  dict, key 与上面 schema 的 parameters 对应。
# - 返回值: 字符串。LLM 会把它当成 tool 消息读, 决定下一步。

# ── get_time ──────────────────────────────────────────────────────────

def _get_time(agent, args):
    now = datetime.datetime.now()
    return now.strftime("现在是 %Y-%m-%d %H:%M:%S (%A)")


# ── calculate ─────────────────────────────────────────────────────────
# 注意: 用 ast 安全求值而不是 eval(), 防止 LLM 喂进恶意代码。

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError(f"不支持的表达式片段: {ast.dump(node)}")


def _calculate(agent, args):
    expr = args["expression"]
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
    except Exception as e:
        return f"无法计算 '{expr}': {e}"
    return f"{expr} = {result}"


# ── note / list_notes ─────────────────────────────────────────────────
# 这两个工具演示"在 agent 实例上保存状态"的常见模式: handler 通过 agent.xxx
# 读写跨工具调用的共享数据。同样的模式可以用于:
#   - 在拾取工具里记下"现在手里有 X", 让放置工具知道
#   - 在导航工具里记下"上一次目标点", 用于 fallback

def _note(agent, args):
    text = args["text"].strip()
    if not text:
        return "笔记为空, 跳过"
    agent.notes.append(text)
    return f"已记下 (第 {len(agent.notes)} 条): {text}"


def _list_notes(agent, args):
    if not agent.notes:
        return "记事本是空的"
    lines = [f"  {i+1}. {n}" for i, n in enumerate(agent.notes)]
    return f"共 {len(agent.notes)} 条笔记:\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  调度表: tool name → handler
# ══════════════════════════════════════════════════════════════════════
#
# 加新工具时除了上面写一个 _xxx 函数, 一定记得在这里加映射。
# 用 dict 而不是一长串 if/elif, 这样加工具是改一行而不是改一片。

_HANDLERS = {
    "get_time": _get_time,
    "calculate": _calculate,
    "note": _note,
    "list_notes": _list_notes,
}


def execute_tool(agent, name: str, args: dict) -> str:
    """工具调度入口。被 ExampleAgent.execute_tool 调用。

    约定:
      - 未知工具 → 返回错误字符串 (而不是 raise), 让 LLM 看到并重试
      - handler 抛异常 → catch 后返回错误字符串, 同样让 LLM 看到
    """
    agent.logger.info(f"执行工具: {name}({args})")
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"未知工具: {name}"
    try:
        return handler(agent, args)
    except Exception as e:
        return f"执行出错: {e}"
