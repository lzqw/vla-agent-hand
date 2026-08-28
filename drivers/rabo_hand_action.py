"""O6 dexterous-hand action executor for the VLA runtime."""

from __future__ import annotations

import time
from typing import Any


class HandActionExecutor:
    """Execute hand actions returned alongside the arm joint action."""

    def __init__(self, devices, *, trace: bool = True) -> None:
        self.devices = devices
        self.trace = trace

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
    def _grasp(hand: Any, action: dict[str, Any]) -> None:
        strength = float(action["strength"])
        fingers = action.get("fingers")
        if fingers is None:
            hand.grasp_force(strength=strength)
        else:
            hand.grasp_force(strength=strength, fingers=[int(v) for v in fingers])

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise TypeError("hand action must be a JSON object")
        action_type = str(action.get("action_type", ""))
        if self.trace:
            print(f"[hand-action] {action_type}", flush=True)

        if action_type == "right_hand_clench":
            self._clench(self.devices.right_hand, action["clench"])
        elif action_type == "left_hand_clench":
            self._clench(self.devices.left_hand, action["clench"])
        elif action_type == "right_hand_grasp_force":
            self._grasp(self.devices.right_hand, action)
        elif action_type == "left_hand_grasp_force":
            self._grasp(self.devices.left_hand, action)
        elif action_type == "wait":
            duration_s = float(action.get("duration_s", 0.0))
            if duration_s < 0.0 or duration_s > 5.0:
                raise ValueError(f"unsafe wait duration {duration_s}")
            time.sleep(duration_s)
        else:
            raise ValueError(f"unsupported hand action_type: {action_type!r}")

        return {"action_type": action_type, "ok": True}
