"""Decode generic VLA actions into the existing Rabo command executor format."""

from __future__ import annotations

from typing import Any


VLA_ACTION_SPACE = "rabo_vla_action_v1"


class VLAActionAdapter:
    """Convert an externally visible VLA action into a Rabo SDK command.

    The 4080 server emits generic observation-to-action outputs.  The web side
    keeps all robot/vendor-specific command semantics behind this adapter.
    """

    @staticmethod
    def _trajectory(value: Any, name: str) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} must be a non-empty list")
        result: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"{name}[{index}] must be an object")
            pose = item.get("pose")
            if not isinstance(pose, (list, tuple)) or len(pose) != 6:
                raise ValueError(f"{name}[{index}].pose must contain 6 values")
            result.append(
                {
                    "label": str(item.get("label", f"{name}[{index}]")),
                    "pose": [float(v) for v in pose],
                }
            )
        return result

    def to_command(self, action: dict[str, Any], action_space: str | None) -> dict[str, Any]:
        if action_space not in {None, VLA_ACTION_SPACE}:
            raise ValueError(f"unsupported VLA action_space: {action_space!r}")
        if not isinstance(action, dict):
            raise TypeError("VLA action must be an object")

        kind = str(action.get("type", ""))

        if kind == "pose_trajectory":
            effector = str(action.get("effector", ""))
            trajectory = self._trajectory(action.get("trajectory"), "trajectory")
            if effector == "right_arm":
                return {"action_type": "right_arm_move_to", "right_moves": trajectory}
            if effector == "left_arm":
                return {"action_type": "left_arm_move_to", "left_moves": trajectory}
            raise ValueError(f"pose_trajectory has invalid effector: {effector!r}")

        if kind == "bimanual_pose_trajectory":
            right_raw = action.get("right_trajectory") or []
            left_raw = action.get("left_trajectory") or []
            if not right_raw and not left_raw:
                raise ValueError("bimanual action requires at least one arm trajectory")
            command: dict[str, Any] = {"action_type": "parallel_arm_sequence"}
            command["right_moves"] = (
                self._trajectory(right_raw, "right_trajectory") if right_raw else []
            )
            command["left_moves"] = (
                self._trajectory(left_raw, "left_trajectory") if left_raw else []
            )
            return command

        if kind == "hand_control":
            effector = str(action.get("effector", ""))
            mode = str(action.get("mode", ""))
            if effector not in {"left_hand", "right_hand"}:
                raise ValueError(f"hand_control has invalid effector: {effector!r}")
            side = "left" if effector == "left_hand" else "right"

            if mode == "clench":
                values = action.get("values")
                if not isinstance(values, list) or len(values) != 6:
                    raise ValueError("hand clench values must contain 6 entries")
                normalized = [None if v is None else float(v) for v in values]
                return {"action_type": f"{side}_hand_clench", "clench": normalized}

            if mode == "grasp_force":
                command = {
                    "action_type": f"{side}_hand_grasp_force",
                    "strength": float(action["strength"]),
                }
                fingers = action.get("fingers")
                if fingers is not None:
                    if not isinstance(fingers, list):
                        raise ValueError("grasp_force fingers must be a list")
                    command["fingers"] = [int(v) for v in fingers]
                return command

            raise ValueError(f"unsupported hand_control mode: {mode!r}")

        if kind == "wait":
            return {"action_type": "wait", "duration_s": float(action.get("duration_s", 0.0))}

        if kind == "done":
            return {"action_type": "done"}

        raise ValueError(f"unsupported VLA action type: {kind!r}")
