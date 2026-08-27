# Agent 开发模板

> 一个**最小可运行**、**LLM 无关**、**机器人无关**的多 agent 系统骨架。
> 拷一份就能开始写你自己的 agent, 也可以作为人类与 AI 编程助手共同遵循的项目约定。

---

## 目录

1. [这是什么](#这是什么)
2. [在平台上跑起来](#在平台上跑起来)
3. [项目结构](#项目结构)
4. [开发约定 (重要, 给 AI 编程助手 / vibe coding 看的也是这一节)](#开发约定)
5. [教程一: 给 agent 加一个新工具](#教程一-给-agent-加一个新工具)
6. [教程二: 写一个新 agent](#教程二-写一个新-agent)
7. [教程三: 换 LLM](#教程三-换-llm)
8. [教程四: 接入硬件 / 机器人](#教程四-接入硬件--机器人)
9. [常见问题](#常见问题)

---

## 这是什么

一个**用 LLM 工具调用 (function calling) 来驱动具体动作**的 agent 框架模板。

它解决三件事:

1. **LLM 循环写一次, 所有 agent 共用** —— 你不必每写一个 agent 就自己实现一遍
   "发请求 → 看有没有工具调用 → 调工具 → 把结果发回去 → 直到 LLM 给出文本回复"。
2. **同一个项目可以承载多个 agent** —— 控制机械臂的 agent、巡检 agent、
   客服 agent 都可以共存于 `agents/` 下, 共享 `core/` 和 `drivers/`。
3. **加新工具与改 prompt 都只改一个文件** —— 让 "扩展 agent 能力" 与 "调
   LLM 规划行为" 这两件最高频的开发活动有明确入口。

模板自带一个不依赖任何硬件的 `example_agent`, 它会的事: 报时、做算术、记笔记。
你看完它就知道怎么写自己的了。

## 在平台上跑起来

这个模板是 rabo 平台的**控制器工程**: 它运行在平台的控制器容器里, 不在你的本机、
也不在浏览器。你只需要**编辑文件 + 填两处配置**, 平台负责安装依赖与启动进程:

1. **配 LLM** —— `agents/example_agent/config.py` 默认已指向 rabo 平台自带的大模型
   API(OpenAI 兼容, 通义千问系列, 支持 function calling, 详见用户手册《22-大模型 API》):
   - `LLM_BASE_URL = "https://ai.rabo.cc/p/qwen"`、`LLM_MODEL = "qwen3.6-flash"`(复杂
     推理/代码可换 `qwen3.6-plus`), 一般无需改。
   - 🔑 **Key 走环境变量 `RABO_LLM_KEY`, 别写进代码**。最省事: 在「个人中心 → API Keys」
     申请一个**【内部使用】**类型的 Key, 平台会把它自动注入到你启动的所有控制器为
     `RABO_LLM_KEY` —— 无需任何手动配置即可用。或在「应用详情页 → 环境变量(ENV)」里
     加 `RABO_LLM_KEY` 覆盖。代码侧 `os.getenv("RABO_LLM_KEY")` 已写好。
   - 想换别的服务(OpenAI / DeepSeek / 自部署等)见「教程三: 换 LLM」。
   - ⚠️ `rabo_dev_kit.Chat` 是纯文本对话、**不支持工具调用**, 不能拿来驱动本框架。

2. **接输入源(可选)** —— 默认没有接任何输入通道(多数用户没有遥控器), `run()` 是个
   占位实现, 启动后只打印一条提示就返回。要让 agent 真正能对话, 需在
   `agents/example_agent/__init__.py` 的 `run()` 里接一个常驻输入源, 最快是取消那段
   `RemoteControl` 示例的注释(需先在场景里建 H5 遥控面板, 并取消 `config.py` 里
   `REMOTE_ID` / `CHAT_CONTROL_ID` 的注释填好), 也可以改成订阅 ROS2 话题。

3. **保存即运行** —— 平台自动按 `requirements.txt` 装依赖, 再跑 `python3 -u main.py`
   (即 `main.py` → `agents/example_agent/run()`)。整个"装/启"过程对你是黑盒。

4. **对话** —— 接好输入源(如上面的 RemoteControl)后, 在对应通道发消息即可。以 H5
   面板 chat 控件为例, 输入下面这些试试:

   ```text
   现在几点
   100 的阶乘是多少   (会失败, 因为 calculate 不支持阶乘 —— 这是好事, 你能学到怎么扩展)
   帮我记一下: 周三开会前先准备 demo
   我都记了什么
   ```

> 进程的日志(每一轮工具调用、结果)会打到平台控制台, 调试看那里。

## 项目结构

```text
template/
├── main.py                    # 入口: 平台无参跑 python3 -u main.py,启动 DEFAULT_AGENT
├── requirements.txt           # 依赖清单 (平台按它装包)
├── README.md                  # 你正在读的文件
├── core/                      # 通用框架代码 (与具体 agent 无关)
│   ├── __init__.py            # 暴露 BaseAgent
│   └── base_agent.py          # ★ LLM 工具调用循环 + JSON 容错
├── drivers/                   # 可复用的硬件 / 感知驱动 (默认空)
│   └── __init__.py            # 解释这里放什么 / 不放什么
└── agents/                    # 各个具体 agent
    ├── __init__.py
    └── example_agent/         # ★ 示例 agent: 报时/算术/笔记 (无硬件)
        ├── __init__.py        # 暴露 run() 入口 (默认占位; 含注释掉的 RemoteControl 接法)
        ├── agent.py           # ExampleAgent 类
        ├── tools.py           # 工具 schema + handler + 调度 (一站式)
        ├── prompts.py         # SYSTEM_PROMPT
        └── config.py          # LLM 配置(默认 rabo 大模型 API)/ 任何常量
```

每个文件**顶部都有详细注释**讲它的角色、约定、扩展点。代码本身就是教程。

## 开发约定

> 这一节既给人类读, 也给 AI 编程助手 (Claude / Cursor / Copilot / …) 读。
> 当你让 AI 帮你改代码时, 让它读一下这一节, 它就知道该往哪儿改。

### 文件归属决策树

| 你想改什么 | 改哪 | 为什么 |
| --- | --- | --- |
| 给 agent 加一个新工具 | `agents/<agent>/tools.py` | schema + handler + 调度都在一个文件, 改一处搞定 |
| 调 prompt / 让 LLM 更会规划 | `agents/<agent>/prompts.py` | 单文件单导出, 调 prompt 唯一入口 |
| 改 LLM 循环行为 (轮数/温度/JSON 容错) | `core/base_agent.py` | 通用框架, 改一次所有 agent 都受益 |
| 换 LLM 模型 / 切 API endpoint | `agents/<agent>/config.py` | base_url / 模型名直接改值 |
| 配 API Key / Token 等敏感信息 | **场景详情页 → 密钥管理** | 运行时注入环境变量, 代码 `os.getenv(...)` 读, 不入仓库 |
| 改硬件 ID / 物理偏移 / agent 私有常量 | `agents/<agent>/config.py` | 跟 agent 绑死的常量都在 config.py |
| 换相机 / 检测器 / 传感器封装 | `drivers/<driver>.py` | 多 agent 共享的硬件抽象 |
| 加一个新 agent | 复制 `agents/example_agent/` 改名 | 模板提供的最小起点 |

### 工具 handler 签名 (硬约定)

```python
def _my_tool(agent, args: dict) -> str:
    ...
```

- 第一个参数永远是 agent 实例 (用它访问硬件 / LLM / 状态)
- 第二个参数永远是 dict (LLM 给的 JSON 参数解析后)
- 返回值永远是 str (LLM 把它当 tool 消息读)
- **捕获异常并 return 错误信息**, 不要 raise (让 LLM 看到错误才能重试)
- 私有函数命名加下划线前缀: `_my_tool` (避免污染 import * 的命名空间)

### Agent 类协议 (硬约定)

子类继承 `BaseAgent` (可同时多继承 ROS Node 等), 在 `__init__` 末尾必须有:

```python
self.llm = OpenAI(base_url=..., api_key=...)
self.model = "..."
self.system_prompt = SYSTEM_PROMPT
self.tools = TOOLS
self.setup_messages()
```

并实现:

```python
def execute_tool(self, name: str, args: dict) -> str:
    return tools_module.execute_tool(self, name, args)
```

外部调用: `agent.run(user_text)` → 字符串回复。

### Agent 子包暴露 run()

`agents/<name>/__init__.py` 必须暴露一个 `run()` 函数, 这是 `main.py` 通过
`importlib.import_module(...).run()` 调用的入口。平台以 `python3 -u main.py` 启动它,
**没有交互式 stdin**, 所以 `run()` 必须常驻并自己接好输入源。模板里 `run()` 默认是
**占位实现**(启动后只打印提示就返回), 你需要选下面一种接好输入源:

- 接 H5 控制面板的 chat 控件 (`rabo_dev_kit.RemoteControl` + `rclpy.spin`) —— **注释掉的示例**, 取消注释即可
- 订阅一个 ROS2 话题作为指令源, 再 spin (纯机器人做法)

⚠️ 不要把 `run()` 写成 `input("you> ")` 这种 stdin REPL —— 在平台容器里会立刻
`EOFError` 退出。

### 不要做的事

- 不要在 `core/` 里 import 任何具体 agent / 机器人 SDK / ROS。框架要 ROS-agnostic。
- 不要把 `_HANDLERS` 里没注册的工具加进 `TOOLS` schema (LLM 会调到一个不存在的 handler)。
- 不要在 handler 里 raise, 用 return 返回错误信息。
- 不要在 `tools.py` 里硬编码 LLM 模型名; 模型名放 `config.py`。
- 不要把 API Key / Token 等敏感信息写死在任何文件里 (会随代码进仓库)；用
  **场景详情页 → 密钥管理** 配成环境变量, 代码侧 `os.getenv(...)` 读取。

## 教程一: 给 agent 加一个新工具

假设你想加一个 `random_number(min, max)` 工具。

**1. 编辑 `agents/example_agent/tools.py`, 在 `TOOLS` 列表加一项:**

```python
{
    "type": "function",
    "function": {
        "name": "random_number",
        "description": "在 [min, max] 范围内返回一个随机整数",
        "parameters": {
            "type": "object",
            "properties": {
                "min": {"type": "integer", "description": "下界 (含)"},
                "max": {"type": "integer", "description": "上界 (含)"},
            },
            "required": ["min", "max"],
        },
    },
},
```

**2. 在同文件里写 handler:**

```python
import random

def _random_number(agent, args):
    lo, hi = args["min"], args["max"]
    if lo > hi:
        return f"参数错误: min({lo}) > max({hi})"
    return str(random.randint(lo, hi))
```

**3. 在 `_HANDLERS` 字典加一行:**

```python
_HANDLERS = {
    ...
    "random_number": _random_number,
}
```

完事。重启 agent, LLM 下次就能用了。

> **可选**: 在 `prompts.py` 里告诉 LLM 何时该用这个工具。tool schema 的
> description 已经够 LLM 用了, 但 prompt 里多嘱咐一句更稳。

## 教程二: 写一个新 agent

```bash
cp -r agents/example_agent agents/my_agent
```

然后修改 4 件事:

1. **`agents/my_agent/config.py`**: 你的 LLM 配置 / 任何硬件常量
2. **`agents/my_agent/prompts.py`**: 你的 SYSTEM_PROMPT (定义 agent 人设与行为)
3. **`agents/my_agent/tools.py`**: 你的工具列表 (按教程一的方法加)
4. **`agents/my_agent/agent.py`**: 把类名 `ExampleAgent` 改成 `MyAgent`,
   增加你需要的状态 / 硬件初始化 / helper 方法

平台默认跑 `python3 -u main.py`(不带参数), 启动 `main.py` 里 `DEFAULT_AGENT` 指定的
agent。所以新建 `my_agent` 后, 要么把 `main.py` 的 `DEFAULT_AGENT` 改成 `"my_agent"`,
要么删掉/改名原 `example_agent`。本地手动调试时也可以 `python3 main.py my_agent` 显式指定。

> **如果你的 agent 要跑在 ROS2 节点里**, 让类多继承 `rclpy.Node`:
>
> ```python
> from rclpy.node import Node
> class MyAgent(BaseAgent, Node):
>     def __init__(self):
>         Node.__init__(self, "my_agent")
>         BaseAgent.__init__(self)
>         self.logger = self.get_logger()  # 用 ROS logger
>         ...
> ```
>
> 然后在 `__init__.py` 的 `run()` 里跑 `rclpy.spin(node)`。
> 接入机器人本体的完整示例见下方「教程四」。

## 教程三: 换 LLM

模板**默认用 rabo 平台自带的大模型 API**(`https://ai.rabo.cc/p/qwen`, 见手册
《22-大模型 API》), Key 走环境变量 `RABO_LLM_KEY`(申请内部 Key 后平台自动注入)。
想换成别的服务: 在 `config.py` 改 `LLM_BASE_URL` / `LLM_MODEL`, 并把
`LLM_API_KEY = os.getenv(...)` 里的环境变量名换成对应服务的(敏感信息走环境变量,
别写进 config.py)。常见组合:

| LLM | LLM_BASE_URL | LLM_MODEL |
| --- | --- | --- |
| **rabo(默认)** | `https://ai.rabo.cc/p/qwen` | `qwen3.6-flash` / `qwen3.6-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` / `gpt-4o` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` / `deepseek-reasoner` |
| Qwen (DashScope) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max` / `qwen-plus` |
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| Ollama (本地) | `http://localhost:11434/v1` | 看你拉了什么 (`qwen2.5:7b` 等) |
| 自部署 vLLM | `http://your-host:8000/v1` | 你部署的模型名 |

> **注意**: 不同 LLM 对 tool calling 的支持质量差别巨大。Qwen / DeepSeek /
> GPT-4o 一般稳, 小型 7B 模型常出非法 JSON 或忘记调工具。模板的
> `safe_parse_args` 能修复一部分非法 JSON, 但不能修复"忘记调工具", 这种
> 问题要在 prompt 里加显式指引。

> **在 rabo 平台上跑时**: 本框架的核心是 **function calling**, 所以必须用支持工具
> 调用的 OpenAI 兼容 LLM。模板默认已配好 rabo 平台大模型 API(qwen3.6 系列, 见手册
> 《22-大模型 API》), Key 走 `RABO_LLM_KEY`(申请内部 Key 后平台自动注入), 开箱即用。
>
> ⚠️ 注意 `rabo_dev_kit` 自带的 `Chat`(平台托管 LLM)**只收发纯文本消息,
> 不支持 function calling**, 因此**不能**直接拿来驱动本框架的工具循环。
> `Chat` 适合"让机器人理解自然语言指令"这类纯对话场景; 要用本模板的工具
> 调用能力, 请用 `config.py` 里默认的 rabo 大模型 API(或其它支持 tool calling 的服务)。

## 教程四: 接入硬件 / 机器人

`drivers/` 默认是空的。模板不强制你用什么硬件 SDK, 你想接什么就接什么。

**典型做法**:

1. 在 `drivers/` 加一个 `my_arm.py`, 写一个类 `MyArm` 把硬件 SDK 包装成你想要的接口
2. 在 `drivers/__init__.py` 暴露: `from .my_arm import MyArm`
3. 在 agent 里:

   ```python
   from drivers import MyArm

   class MyAgent(BaseAgent):
       def __init__(self):
           super().__init__()
           self.arm = MyArm(robot_id=config.ROBOT_ID)
           ...
   ```

4. 在 `tools.py` 写工具 handler 直接用 `agent.arm.move_to(...)` 即可

**在 rabo 平台上接机器人**: 平台的机器人控制 SDK 是 `rabo_robocap`(机械臂 /
底盘 / 夹爪)和工具包 `rabo_dev_kit`(虚实同步桥 / 远程遥控 / 数据记录)。
直接在 agent 里组合即可,例如:

```python
from rabo_robocap import UR5, RobotiqHandEGripper

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # robot_id 必须填场景 SDF 里该 model 的 @name
        self.arm = UR5(robot_id="<sdf-model-name>", mode="sim")
        self.gripper = RobotiqHandEGripper(robot_id="<sdf-model-name>", mode="sim")
        ...  # 接着设好 self.llm / self.model / self.tools 并 setup_messages()
```

然后在 `tools.py` 的 handler 里调 `agent.arm.move_to(...)` / `agent.gripper.grasp()`。
注意:用到这些包时,务必把它们加进 `requirements.txt`(见下)。

## 常见问题

**Q: 工具调了但 LLM 没拿结果继续**
A: 先看 log 里 `[agent] 工具结果: ...` 是不是真的写进去了。如果工具 return
了 None, LLM 会困惑 — 永远 return 字符串。

**Q: 服务端 400, 报 message 格式错误**
A: 多半是上一轮 tool_call.arguments 是非法 JSON, 中毒了。模板的
`safe_parse_args` 应该已经兜底, 如果还出问题可以在 `core/base_agent.py` 加
更激进的修复策略。

**Q: 怎么持久化对话历史 / 重启后恢复?**
A: 模板没做这一层。需要的话在 `BaseAgent.run()` 末尾把 `self.messages` 序列化
到磁盘, 在 `__init__` 末尾反序列化进来。
⚠️ 关键坑: `self.messages` 里**混了两种类型** —— 用户/工具消息是普通 dict,
而 LLM 回复那条是 SDK 返回的对象 (`ChatCompletionMessage`, 其 `tool_calls`
元素是 `ChatCompletionMessageToolCall`)。直接 `json.dumps` 会失败。两种做法:
(1) append 前先用 `msg.model_dump()` 把 SDK 对象转成 dict 再入历史(最省事);
(2) 持久化时对这两类分别处理。无论哪种, 反序列化后 `tool_call_id` 必须与
对应的 assistant 消息严格配对, 否则下一轮请求会被服务端 400 拒绝。

**Q: 多个 LLM 接力 / 多 agent 协同怎么做?**
A: 当前 `BaseAgent` 只考虑单 LLM 单 agent。如果要做"主 agent 把任务分派给
子 agent" 的 multi-agent, 把"调用子 agent" 也写成一个工具就行 — 在 handler
里调 `subordinate_agent.run(...)` 拿回字符串。

**Q: 工具有状态 (比如夹爪开合度) 怎么共享?**
A: 放 `self` 上。`example_agent` 的 `self.notes` 就是这种用法。任何工具
handler 都拿得到 `agent`, 直接读写就行。

---

这套结构在真实项目里已经过验证: 用同样的 `core/` + `drivers/` + `agents/`
分层, 可以搭出 "UR5 机械臂 + RGBD 相机 + 视觉检测 + LLM" 这种完整 agent ——
把相机/检测器封装进 `drivers/`, 把机械臂动作封装成 `tools.py` 里的工具,
agent 类同时继承 `BaseAgent` 与 `rclpy.Node` 即可。
