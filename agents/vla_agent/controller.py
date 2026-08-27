"""Rabo -> 4080 remote closed-loop controller."""

from __future__ import annotations

import base64
import json
import signal
import threading
import time
import uuid
from typing import Any

import cv2
import numpy as np

from drivers import (
    MockSensorBackend,
    RaboDevices,
    RemoteCommandExecutor,
    Ros2SensorBackend,
    VLA_ACTION_SPACE,
    VLAActionAdapter,
    load_config,
)

from . import config as runtime
from .remote_client import RemoteVLAClient


class RemoteVLAController:
    def __init__(self) -> None:
        if not runtime.ROBOT_CONFIG_PATH.is_file():
            raise FileNotFoundError(f"Robot config not found: {runtime.ROBOT_CONFIG_PATH}")
        if runtime.STARTUP_DELAY_S:
            time.sleep(runtime.STARTUP_DELAY_S)

        self.stop_event = threading.Event()
        self.robot_config = load_config(runtime.ROBOT_CONFIG_PATH)
        self.camera_names = tuple(runtime.CAMERA_NAMES)
        missing = [
            name for name in self.camera_names
            if name not in self.robot_config.camera_topics
        ]
        if missing:
            raise RuntimeError(f"Unknown VLA cameras in config: {missing}")

        token = RemoteVLAClient.load_token(runtime.TOKEN_ENV, runtime.TOKEN_FILE)
        if not token:
            raise RuntimeError(
                "Missing 4080 token. Configure VLA_TOKEN or put the token in "
                f"{runtime.TOKEN_FILE}."
            )
        self.remote = RemoteVLAClient(
            ws_url=runtime.WS_URL,
            http_url=runtime.HTTP_URL,
            health_url=runtime.HEALTH_URL,
            token=token,
            transport=runtime.TRANSPORT,
            ws_auth_mode=runtime.WS_AUTH_MODE,
            timeout_s=runtime.NETWORK_TIMEOUT_S,
        )

        self.devices: RaboDevices | None = None
        self.command_executor: RemoteCommandExecutor | None = None
        self.sensors: Any | None = None
        self.action_adapter = VLAActionAdapter()

        self.session_id = uuid.uuid4().hex[:8]
        self.instruction = runtime.INSTRUCTION or str(self.robot_config.dataset["task"])
        self.last_action: dict[str, Any] | None = None
        self.last_command: dict[str, Any] | None = None
        self.last_execution: dict[str, Any] | None = None

    @staticmethod
    def _json(event: str, **values: Any) -> None:
        print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)

    def _setup_local_runtime(self) -> None:
        if runtime.SENSOR_BACKEND == "mock":
            self.sensors = MockSensorBackend(self.robot_config)
            return
        self.devices = RaboDevices(self.robot_config, mode=runtime.MODE)
        self.command_executor = RemoteCommandExecutor(self.devices, trace=True)
        self.sensors = Ros2SensorBackend(
            self.robot_config,
            joint_reader=self.devices.read_full_state,
            abort_event=self.stop_event,
        )

    def _request_stop(self, signum: int, _frame: Any) -> None:
        self._json("signal", signum=signum)
        self.stop_event.set()
        if self.sensors is not None:
            try:
                self.sensors.wake()
            except Exception:
                pass

    @staticmethod
    def _jpeg_b64(image_bgr: np.ndarray) -> tuple[str, int, int]:
        ok, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), runtime.JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        height, width = image_bgr.shape[:2]
        return base64.b64encode(encoded.tobytes()).decode("ascii"), width, height

    def _payload(self, snapshot: Any, request_id: str, step: int) -> dict[str, Any]:
        images: dict[str, dict[str, Any]] = {}
        for name in self.camera_names:
            data, width, height = self._jpeg_b64(snapshot.images[name])
            images[name] = {
                "encoding": "jpeg_base64",
                "mime_type": "image/jpeg",
                "width": width,
                "height": height,
                "data": data,
            }

        payload: dict[str, Any] = {
            "type": "state",
            "protocol": runtime.ORACLE_PROTOCOL,
            "request_id": request_id,
            "episode_id": self.session_id,
            "step": int(step),
            "instruction": self.instruction,
            "state": np.asarray(snapshot.state, dtype=np.float32).tolist(),
            "images": images,
            "camera_reused": {
                name: bool(snapshot.camera_reused.get(name, False))
                for name in self.camera_names
            },
            "image_stamps": {
                name: float(snapshot.image_stamps.get(name, 0.0))
                for name in self.camera_names
            },
            "sim_time": float(snapshot.sim_time),
            "previous_action": self.last_action,
            # Retain this compatibility field while the hybrid backend still
            # uses the validated command executor internally.
            "previous_command": self.last_command,
            "previous_execution": self.last_execution,
        }
        if runtime.SEND_FULL_STATE:
            payload["full_state"] = np.asarray(
                snapshot.full_state, dtype=np.float32
            ).tolist()
        return payload

    @staticmethod
    def _remote_action(reply: dict[str, Any]) -> np.ndarray:
        value = reply.get("action")
        if value is None:
            chunk = reply.get("action_chunk")
            if isinstance(chunk, list) and chunk:
                value = chunk[0]
        if value is None:
            raise RuntimeError("Remote response has no executable action")
        action = np.asarray(value, dtype=np.float32)
        if action.ndim != 1 or action.size == 0 or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid remote action shape/value: {action.shape}")
        return action

    def _execute_numeric(self, action: np.ndarray, snapshot: Any) -> dict[str, Any] | None:
        if not runtime.EXECUTE_ACTIONS:
            return {"shadow_numeric": True, "action_dim": int(action.size)}
        if self.devices is None:
            raise RuntimeError("Cannot actuate numeric action without real Rabo devices")

        current = np.asarray(snapshot.state, dtype=np.float32)
        if action.shape == (26,):
            target = action
            scope = "full"
        elif action.shape == (14,):
            target = current.copy()
            target[:14] = action
            scope = "arms"
        elif action.shape == (7,):
            if runtime.ACTION_MODE not in {"left7", "right7"}:
                raise RuntimeError("7D action requires VLA_ACTION_MODE=left7/right7")
            target = current.copy()
            if runtime.ACTION_MODE == "left7":
                target[:7] = action
            else:
                target[7:14] = action
            scope = "arms"
        else:
            raise RuntimeError(f"Unsupported executable action shape {action.shape}")

        return self.devices.command_action(
            action=target,
            current_full_state=snapshot.full_state,
            scope=scope,
            max_arm_step=runtime.MAX_ARM_STEP_RAD,
            max_hand_step=runtime.MAX_HAND_STEP_RAD,
        )

    def _execute_structured_command(self, command: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        self.last_command = command
        if str(command.get("action_type")) == "done":
            self.last_execution = {"done": True}
            return True, self.last_execution
        if not runtime.EXECUTE_REMOTE_COMMANDS:
            self.last_execution = {"shadow": True}
            return False, self.last_execution
        if self.command_executor is None:
            raise RuntimeError("Cannot execute remote action with mock backend")
        started = time.perf_counter()
        result = self.command_executor.execute(command)
        result["elapsed_s"] = round(time.perf_counter() - started, 6)
        self.last_execution = result
        return bool(result.get("done", False)), result

    def _execute_reply(
        self, reply: dict[str, Any], snapshot: Any, request_id: str
    ) -> tuple[bool, dict[str, Any] | None]:
        reply_request_id = reply.get("request_id")
        if reply_request_id not in {None, request_id}:
            raise RuntimeError(
                f"4080 request_id mismatch: expected {request_id!r}, got {reply_request_id!r}"
            )

        protocol = reply.get("protocol")
        if protocol not in {None, runtime.ORACLE_PROTOCOL}:
            raise RuntimeError(f"Unexpected remote action protocol: {protocol!r}")

        action_value = reply.get("action")
        if isinstance(action_value, dict):
            action_space = reply.get("action_space")
            if action_space not in {None, VLA_ACTION_SPACE}:
                raise RuntimeError(f"Unexpected VLA action_space: {action_space!r}")
            self.last_action = action_value
            command = self.action_adapter.to_command(action_value, action_space)
            return self._execute_structured_command(command)

        # Backward-compatible structured command path for the previous bridge.
        command = reply.get("command")
        if isinstance(command, dict):
            self.last_action = None
            return self._execute_structured_command(command)

        action = self._remote_action(reply)
        result = self._execute_numeric(action, snapshot)
        self.last_execution = result
        return False, result

    @staticmethod
    def _reply_kind(reply: dict[str, Any]) -> tuple[str, str | None]:
        action = reply.get("action")
        if isinstance(action, dict):
            return "vla_action", str(action.get("type", "unknown"))
        command = reply.get("command")
        if isinstance(command, dict):
            return "legacy_command", str(command.get("action_type", "unknown"))
        return "numeric_action", None

    def run(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._request_stop)
            except (ValueError, OSError):
                pass

        self._json(
            "vla_agent_startup",
            ws_url=runtime.WS_URL,
            http_url=runtime.HTTP_URL,
            health_url=runtime.HEALTH_URL,
            transport=runtime.TRANSPORT,
            execute_commands=runtime.EXECUTE_REMOTE_COMMANDS,
            execute_numeric=runtime.EXECUTE_ACTIONS,
            reset_fixed_scene=runtime.RESET_FIXED_SCENE,
            local_preposition=runtime.LOCAL_PREPOSITION,
            cameras=list(self.camera_names),
        )

        dt = 1.0 / runtime.CONTROL_HZ
        step = 0
        try:
            health = self.remote.health()
            self._json("remote_health_ok", health=health)
            active_transport = self.remote.connect()
            self._json("remote_transport_ready", transport=active_transport)

            self._setup_local_runtime()
            assert self.sensors is not None
            self.sensors.start()
            self.sensors.wait_ready(runtime.READY_TIMEOUT_S)
            self._json("sensors_ready")

            if self.devices is not None and runtime.EXECUTE_REMOTE_COMMANDS:
                if runtime.RESET_FIXED_SCENE:
                    self._json("fixed_scene_reset_start")
                    self.devices.reset_fixed_scene()
                    self._json("fixed_scene_reset_done")
                if runtime.LOCAL_PREPOSITION:
                    self._json("pre_position_start")
                    self.devices.pre_position()
                    self._json("pre_position_done")

            self.sensors.reset_clock()
            while not self.stop_event.is_set():
                snapshot = self.sensors.sample_next(dt, dt, self.stop_event)
                if snapshot is None:
                    continue

                request_id = f"{self.session_id}-step-{step}"
                payload = self._payload(snapshot, request_id, step)
                started = time.perf_counter()
                remote_reply = self.remote.infer(payload)
                rtt_ms = (time.perf_counter() - started) * 1000.0
                done, result = self._execute_reply(
                    remote_reply.payload, snapshot, request_id
                )
                response_kind, action_type = self._reply_kind(remote_reply.payload)

                self._json(
                    "remote_reply",
                    request_id=request_id,
                    step=step,
                    transport=remote_reply.transport,
                    response_kind=response_kind,
                    action_type=action_type,
                    action_space=remote_reply.payload.get("action_space"),
                    policy=remote_reply.payload.get("policy"),
                    model=remote_reply.payload.get("model"),
                    backend=remote_reply.payload.get("backend"),
                    inference_ms=remote_reply.payload.get("inference_ms"),
                    rtt_ms=round(rtt_ms, 3),
                    observation_check=remote_reply.payload.get("observation_check"),
                    execution=result,
                )

                if done:
                    self._json("remote_done", step=step)
                    return
                step += 1
                if runtime.MAX_CYCLES and step >= runtime.MAX_CYCLES:
                    self._json("max_cycles_reached", steps=step)
                    return
        finally:
            self.stop_event.set()
            if self.sensors is not None:
                try:
                    self.sensors.close()
                except Exception as exc:
                    self._json("sensor_close_warning", error=repr(exc))
            self.remote.close()
            if self.devices is not None:
                if runtime.EXECUTE_ACTIONS or runtime.EXECUTE_REMOTE_COMMANDS:
                    self.devices.stop()
                self.devices.shutdown()
            self._json("vla_agent_stopped", steps=step)
