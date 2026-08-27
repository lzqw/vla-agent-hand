"""Execute structured remote commands with the official Rabo SDK."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class RemoteCommandExecutor:
    def __init__(self, devices, *, trace: bool = True) -> None:
        self.devices = devices
        self.trace = trace

    @staticmethod
    def _pose6(value: Any) -> tuple[float, float, float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError(f"pose must be length 6, got {value!r}")
        return tuple(float(v) for v in value)  # type: ignore[return-value]

    @staticmethod
    def _run_parallel(*functions: Callable[[], None]) -> None:
        errors: list[BaseException] = []
        lock = threading.Lock()

        def invoke(fn: Callable[[], None]) -> None:
            try:
                fn()
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=invoke, args=(fn,)) for fn in functions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("parallel remote arm command failed") from errors[0]

    @staticmethod
    def _move_to(arm: Any, move: dict[str, Any]) -> None:
        label = str(move.get("label", "remote arm move"))
        x, y, z, roll, pitch, yaw = RemoteCommandExecutor._pose6(move["pose"])
        for attempt in range(2):
            result = arm.move_to(x, y, z, roll=roll, pitch=pitch, yaw=yaw)
            if result is not False:
                return
            print(
                f"[remote-command] {label} attempt {attempt + 1} returned False",
                flush=True,
            )
            if attempt == 0:
                time.sleep(0.10)
        raise RuntimeError(f"{label}: move_to returned False twice")

    @staticmethod
    def _clench(hand: Any, values: Any) -> None:
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise ValueError("clench must contain 6 values")
        hand.clench(
            thumb_rotation=values[0],
            thumb_bend=values[1],
            index=values[2],
            middle=values[3],
            ring=values[4],
            pinky=values[5],
        )

    @staticmethod
    def _grasp(hand: Any, command: dict[str, Any]) -> None:
        strength = float(command["strength"])
        fingers = command.get("fingers")
        if fingers is None:
            hand.grasp_force(strength=strength)
        else:
            hand.grasp_force(strength=strength, fingers=[int(v) for v in fingers])

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(command, dict):
            raise TypeError("remote command must be a JSON object")
        action_type = str(command.get("action_type", ""))
        if self.trace:
            print(f"[remote-command] {action_type}", flush=True)

        if action_type == "right_arm_move_to":
            moves = command.get("right_moves") or []
            if not moves:
                raise ValueError("right_arm_move_to requires right_moves")
            for move in moves:
                self._move_to(self.devices.right_arm, move)

        elif action_type == "left_arm_move_to":
            moves = command.get("left_moves") or []
            if not moves:
                raise ValueError("left_arm_move_to requires left_moves")
            for move in moves:
                self._move_to(self.devices.left_arm, move)

        elif action_type == "parallel_arm_sequence":
            right_moves = list(command.get("right_moves") or [])
            left_moves = list(command.get("left_moves") or [])
            if not right_moves and not left_moves:
                raise ValueError("parallel_arm_sequence requires at least one side")

            def right() -> None:
                for move in right_moves:
                    self._move_to(self.devices.right_arm, move)

            def left() -> None:
                for move in left_moves:
                    self._move_to(self.devices.left_arm, move)

            self._run_parallel(right, left)

        elif action_type == "right_hand_clench":
            self._clench(self.devices.right_hand, command["clench"])

        elif action_type == "left_hand_clench":
            self._clench(self.devices.left_hand, command["clench"])

        elif action_type == "right_hand_grasp_force":
            self._grasp(self.devices.right_hand, command)

        elif action_type == "left_hand_grasp_force":
            self._grasp(self.devices.left_hand, command)

        elif action_type == "wait":
            duration_s = float(command.get("duration_s", 0.0))
            if duration_s < 0.0 or duration_s > 5.0:
                raise ValueError(f"unsafe wait duration {duration_s}")
            time.sleep(duration_s)

        elif action_type == "done":
            return {"done": True}

        else:
            raise ValueError(f"unsupported remote action_type: {action_type!r}")

        return {"done": False, "action_type": action_type}
