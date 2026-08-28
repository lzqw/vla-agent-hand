"""Closed-loop VLA controller for the Rabo simulation runtime."""

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
    HandActionExecutor,
    MockSensorBackend,
    RaboDevices,
    Ros2SensorBackend,
    load_config,
)

from . import config as runtime
from .remote_client import RemoteVLAClient


class VLAController:
    """Capture multimodal observations, request actions, and execute them."""

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
                "Missing VLA token. Configure VLA_TOKEN or put the token in "
                f"{runtime.TOKEN_FILE}."
            )
        self.policy = RemoteVLAClient(
            ws_url=runtime.WS_URL,
            http_url=runtime.HTTP_URL,
            health_url=runtime.HEALTH_URL,
            token=token,
            transport=runtime.TRANSPORT,
            ws_auth_mode=runtime.WS_AUTH_MODE,
            timeout_s=runtime.NETWORK_TIMEOUT_S,
            client_name="rabo-vla-runtime",
        )

        self.devices: RaboDevices | None = None
        self.hand_executor: HandActionExecutor | None = None
        self.sensors: Any | None = None

        self.episode_id = uuid.uuid4().hex[:8]
        self.instruction = runtime.INSTRUCTION or str(self.robot_config.dataset["task"])

    @staticmethod
    def _json(event: str, **values: Any) -> None:
        print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)

    def _setup_runtime(self) -> None:
        if runtime.SENSOR_BACKEND == "mock":
            self.sensors = MockSensorBackend(self.robot_config)
            return
        self.devices = RaboDevices(self.robot_config, mode=runtime.MODE)
        self.hand_executor = HandActionExecutor(self.devices, trace=True)
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

    def _observation(self, snapshot: Any, request_id: str) -> dict[str, Any]:
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

        return {
            "type": "state",
            "protocol": runtime.PROTOCOL,
            "request_id": request_id,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "state": np.asarray(snapshot.state, dtype=np.float32).tolist(),
            "full_state": np.asarray(snapshot.full_state, dtype=np.float32).tolist(),
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
        }

    @staticmethod
    def _action_vector(reply: dict[str, Any]) -> np.ndarray:
        value = reply.get("action")
        action = np.asarray(value, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"VLA action must contain 14 finite joint targets, got {action.shape}"
            )
        return action

    def _execute_arm_action(self, action: np.ndarray, snapshot: Any) -> dict[str, Any]:
        if not runtime.EXECUTE_ACTIONS:
            return {"shadow": True, "action_dim": 14}
        if self.devices is None:
            raise RuntimeError("Cannot actuate VLA action without real Rabo devices")
        return self.devices.command_arm_joint_action(
            action=action,
            current_full_state=snapshot.full_state,
            max_arm_step=runtime.MAX_ARM_STEP_RAD,
        )

    def _execute_hand_action(self, hand_action: Any) -> dict[str, Any] | None:
        if hand_action is None:
            return None
        if not isinstance(hand_action, dict):
            raise RuntimeError("hand_command must be an object when present")
        if not runtime.EXECUTE_HAND_ACTIONS:
            return {"shadow": True, "action_type": hand_action.get("action_type")}
        if self.hand_executor is None:
            raise RuntimeError("Cannot actuate hand action without real Rabo devices")
        started = time.perf_counter()
        result = self.hand_executor.execute(hand_action)
        result["elapsed_s"] = round(time.perf_counter() - started, 6)
        return result

    def _execute_reply(
        self, reply: dict[str, Any], snapshot: Any, request_id: str
    ) -> tuple[bool, dict[str, Any]]:
        if reply.get("type") not in {None, "action"}:
            raise RuntimeError(f"Unexpected VLA response type: {reply.get('type')!r}")
        if reply.get("request_id") not in {None, request_id}:
            raise RuntimeError(
                f"VLA request_id mismatch: expected {request_id!r}, "
                f"got {reply.get('request_id')!r}"
            )
        if reply.get("protocol") not in {None, runtime.PROTOCOL}:
            raise RuntimeError(f"Unexpected VLA protocol: {reply.get('protocol')!r}")
        if reply.get("action_space") != runtime.ACTION_SPACE:
            raise RuntimeError(
                f"Expected action_space={runtime.ACTION_SPACE!r}, "
                f"got {reply.get('action_space')!r}"
            )

        action = self._action_vector(reply)
        arm_result = self._execute_arm_action(action, snapshot)
        hand_action = reply.get("hand_command")
        hand_result = self._execute_hand_action(hand_action)

        execution: dict[str, Any] = {"arm": arm_result}
        if hand_result is not None:
            execution["hand"] = hand_result
        return bool(reply.get("done", False)), execution

    @staticmethod
    def _validate_policy_health(health: dict[str, Any]) -> None:
        if health.get("status") != "ok" or not bool(health.get("model_loaded", False)):
            raise RuntimeError(f"VLA policy is not ready: {health}")
        if health.get("action_space") != runtime.ACTION_SPACE:
            raise RuntimeError(
                f"VLA action space mismatch: expected {runtime.ACTION_SPACE!r}, "
                f"got {health.get('action_space')!r}"
            )
        if int(health.get("output_action_dim", -1)) != 14:
            raise RuntimeError(
                f"VLA output_action_dim must be 14, got {health.get('output_action_dim')!r}"
            )

    def run(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._request_stop)
            except (ValueError, OSError):
                pass

        self._json(
            "vla_startup",
            transport=runtime.TRANSPORT,
            action_space=runtime.ACTION_SPACE,
            execute_actions=runtime.EXECUTE_ACTIONS,
            execute_hand_actions=runtime.EXECUTE_HAND_ACTIONS,
            cameras=list(self.camera_names),
        )

        dt = 1.0 / runtime.CONTROL_HZ
        cycle = 0
        try:
            health = self.policy.health()
            self._validate_policy_health(health)
            self._json(
                "vla_ready",
                model=health.get("model"),
                action_space=health.get("action_space"),
                output_action_dim=health.get("output_action_dim"),
            )
            active_transport = self.policy.connect()
            self._json("policy_transport_ready", transport=active_transport)

            self._setup_runtime()
            assert self.sensors is not None
            self.sensors.start()
            self.sensors.wait_ready(runtime.READY_TIMEOUT_S)
            self._json("sensors_ready")

            if self.devices is not None and (
                runtime.EXECUTE_ACTIONS or runtime.EXECUTE_HAND_ACTIONS
            ):
                if runtime.RESET_FIXED_SCENE:
                    self.devices.reset_fixed_scene()
                    self._json("scene_reset_done")
                if runtime.LOCAL_PREPOSITION:
                    self.devices.pre_position()
                    self._json("pre_position_done")

            self.sensors.reset_clock()
            while not self.stop_event.is_set():
                snapshot = self.sensors.sample_next(dt, dt, self.stop_event)
                if snapshot is None:
                    continue

                request_id = f"{self.episode_id}-{cycle:06d}"
                observation = self._observation(snapshot, request_id)
                started = time.perf_counter()
                remote_reply = self.policy.infer(observation)
                rtt_ms = (time.perf_counter() - started) * 1000.0
                done, execution = self._execute_reply(
                    remote_reply.payload, snapshot, request_id
                )
                hand_action = remote_reply.payload.get("hand_command")

                self._json(
                    "vla_action",
                    cycle=cycle,
                    transport=remote_reply.transport,
                    model=remote_reply.payload.get("model"),
                    action_space=remote_reply.payload.get("action_space"),
                    action_dim=14,
                    hand_action_type=(hand_action or {}).get("action_type")
                    if isinstance(hand_action, dict)
                    else None,
                    inference_ms=remote_reply.payload.get("inference_ms"),
                    rtt_ms=round(rtt_ms, 3),
                    execution=execution,
                    done=done,
                )

                if done:
                    self._json("episode_done", cycles=cycle + 1)
                    return
                cycle += 1
                if runtime.MAX_CYCLES and cycle >= runtime.MAX_CYCLES:
                    self._json("max_cycles_reached", cycles=cycle)
                    return
        finally:
            self.stop_event.set()
            if self.sensors is not None:
                try:
                    self.sensors.close()
                except Exception as exc:
                    self._json("sensor_close_warning", error=repr(exc))
            self.policy.close()
            if self.devices is not None:
                if runtime.EXECUTE_ACTIONS or runtime.EXECUTE_HAND_ACTIONS:
                    self.devices.stop()
                self.devices.shutdown()
            self._json("vla_stopped", cycles=cycle)
