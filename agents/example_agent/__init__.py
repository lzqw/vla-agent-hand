"""ExampleAgent 子包入口。

main.py 会通过 importlib 加载本包, 然后调用 run()。
所以每个 agent 子包必须在 __init__.py 暴露一个 run() 函数。
"""

import logging

from .agent import ExampleAgent

__all__ = ["ExampleAgent", "run"]


def run():
    """ExampleAgent 的运行入口 —— 在 rabo 平台控制器容器里被 `python3 -u main.py` 启动。

    当前实现: 实例化 ExampleAgent 后返回。模板默认保持最小可运行 —— 不接入 LLM、也不接
    输入通道, 所以 agent 此时不处理任何消息。要让它真正对话:
      ① 接入 LLM: 在 config.py 填好 LLM 配置(并配好 RABO_LLM_KEY), 取消 agent.py 里 LLM
         接入代码及相关 import 的注释;
      ② 接入输入源: 平台容器没有交互式 stdin, 取消下方 RemoteControl 示例的注释并删掉
         `return` 即可(也可自行订阅 ROS2 话题再 rclpy.spin)。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("ExampleAgent")

    # 实例化 ExampleAgent(其 __init__ 会打印就绪日志)。
    agent = ExampleAgent()
    log.info(f"{type(agent).__name__} 启动成功。")
    return

    # ──────────────────────────────────────────────────────────────────
    # 可选输入示例: H5 控制面板 chat 控件 (RemoteControl)
    #
    # 用户在场景的 H5 控制面板 chat 控件里输入文字, 经 WebSocket → 回调送到这里;
    # agent 的回复再用 rc.send(...) 回传到同一个 chat 控件显示。进程靠 rclpy.spin 常驻。
    #
    # 启用前:
    #   1. 在场景里创建一个 H5 遥控面板, 拿到面板 ID, 取消 config.py 里 REMOTE_ID /
    #      CHAT_CONTROL_ID 的注释并填好;
    #   2. 启用 LLM: 取消 config.py 里 LLM_* 的注释 + agent.py 里 LLM 接入代码及相关 import
    #      的注释, 并配好 RABO_LLM_KEY(申请内部 Key 后平台自动注入, 见 config.py);
    #   3. 删掉上面的 `return`, 并取消下面整段的注释。
    #
    # 若你的 agent 还要控制机器人 (rabo_robocap), 在 ExampleAgent.__init__ 里初始化
    # 即可 —— 那些 SDK 自带后台 executor, 不影响这里只 spin RemoteControl 这一个节点。
    # ──────────────────────────────────────────────────────────────────
    #
    # import sys
    #
    # import rclpy
    # from rabo_dev_kit import RemoteControl
    #
    # from . import config
    #
    # # 启用前先校验必填配置(LLM key / 面板 ID),缺失就给出明确提示并退出,
    # # 而不是等运行到一半才发现没配。
    # missing = [k for k in ("LLM_API_KEY", "REMOTE_ID") if not getattr(config, k, "")]
    # if missing:
    #     log.error(
    #         "缺少必填配置：%s。LLM_API_KEY 来自环境变量 RABO_LLM_KEY（申请内部 Key 后"
    #         "平台自动注入，见 config.py）；REMOTE_ID 在 agents/example_agent/config.py 填写。",
    #         " / ".join(missing),
    #     )
    #     sys.exit(1)
    #
    # rclpy.init()
    # # agent 已在上面实例化,这里直接用
    #
    # # rc 在回调里要用, 但要等创建后才有; 用一个 holder 避免闭包写入问题。
    # holder = {"rc": None}
    #
    # def on_control(data: dict):
    #     """RemoteControl 回调。data 形如 {'chat-1': {'value': '用户输入'}, ...}。
    #
    #     注意: 该回调在 ROS executor 线程里执行, agent.run() 是阻塞的 (LLM + 工具
    #     可能耗时数秒), 期间不会处理新的面板消息 —— 对单用户对话足够, 高并发需另做。
    #     """
    #     text = data.get(config.CHAT_CONTROL_ID, {}).get("value")
    #     if not text:
    #         return  # 这条面板事件不是 chat 输入 (可能是按钮/滑块), 忽略
    #
    #     agent.logger.info(f"[panel] 收到: {text}")
    #     try:
    #         reply = agent.run(str(text))
    #     except Exception as e:
    #         reply = f"处理出错: {e}"
    #
    #     rc = holder["rc"]
    #     if rc is not None:
    #         try:
    #             rc.send(config.CHAT_CONTROL_ID, reply)
    #         except Exception as e:
    #             agent.logger.error(f"[panel] 回传失败: {e}")
    #
    # rc = RemoteControl(remote_id=config.REMOTE_ID, callback=on_control)
    # holder["rc"] = rc
    # agent.logger.info(
    #     f"ExampleAgent 已就绪, 监听控制面板 {config.REMOTE_ID} 的 "
    #     f"{config.CHAT_CONTROL_ID} 控件。"
    # )
    #
    # try:
    #     rclpy.spin(rc)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     rc.destroy_node()
    #     rclpy.try_shutdown()
