"""ExampleAgent 的全部常量与配置。

────────────────────────────────────────────────────────────────────────
配置分两类：

1. 普通配置（base_url / 模型名 / 控制面板 ID 等）——直接改本文件的值即可。

2. 敏感信息（API Key、Token、密码等）——**用环境变量**，不要写死在本文件里。
   原因：config.py 会随控制器代码提交到 Git 仓库，复制/分享场景时一并带走，
   明文密钥极易泄漏。

LLM 采用 OpenAI 兼容协议。本框架靠 **function calling**，必须选支持工具调用的模型。
默认指向 rabo 平台自带的大模型 API（OpenAI 兼容，通义千问系列支持工具调用），
详见用户手册《22-大模型 API》。也可改成 OpenAI / DeepSeek / 自部署 vLLM 等其它
兼容服务，见 README「教程三：换 LLM」。
────────────────────────────────────────────────────────────────────────
"""

import os

# ── LLM 配置（接入大模型时启用，默认走 rabo 平台大模型 API，详见手册 22-llm-api）──
# 模板默认保持最小可运行、不接 LLM。接入时取消下面三行注释（并同步取消 agent.py 里
# LLM 接入代码及相关 import 的注释）：
#   - base_url / 模型名直接填。可用模型：qwen3.6-flash（日常/高频，快且省）、
#     qwen3.6-plus（复杂推理/代码生成）。
#   - Key 走环境变量 RABO_LLM_KEY，不要把真实 key 写进代码（会随仓库/分享场景泄漏）。
#     最省事：在「个人中心 → API Keys」申请一个【内部使用】类型的 Key，平台会自动注入到
#     你启动的所有控制器为 RABO_LLM_KEY 环境变量；或在「应用详情页 → 环境变量(ENV)」里
#     加 RABO_LLM_KEY = sk-rabo-xxx（应用级覆盖）。
#   - 换别的服务：把 base_url / model 换成对应值，把 RABO_LLM_KEY 换成对应服务的 key 环境变量名。
# LLM_BASE_URL = "https://ai.rabo.cc/p/qwen"
# LLM_MODEL = "qwen3.6-flash"
# LLM_API_KEY = os.getenv("RABO_LLM_KEY", "")

# ── 平台控制面板（可选输入通道，默认关闭）──
# 默认没有遥控器，所以这两项先注释掉。若你在场景里创建了 H5 遥控面板、想用它的
# chat 控件作为 agent 的输入/输出通道，再取消下面两行的注释并填入面板 ID，同时
# 取消 __init__.py 里 run() 中 RemoteControl 那段示例的注释。
# REMOTE_ID:       H5 控制面板 ID（在场景的控制面板设置里获取）。
# CHAT_CONTROL_ID: 面板上 chat 控件的 ID，用户在此输入、agent 回复也回传到这里。
# RemoteControl 连接所需的 RABO_REMOTE_WS_URL 由平台运行环境自动注入，无需在此配置。
# REMOTE_ID = ""            # ← 例如 "abc123"
# CHAT_CONTROL_ID = "chat-1"
