from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any


logger = logging.getLogger("uvicorn.error")

RABO_PROTOCOL = "rabo_command_v1"
REQUIRED_CAMERAS = frozenset(
    {"cam_high", "cam_left_wrist", "cam_right_wrist"}
)

# Fixed simulation-scene ground truth.  These Cartesian poses are consumed by
# the browser-side Rabo SDK; this service intentionally performs no A7 IK.
B = [-0.3413, -0.1710, 0.2806, 0.0, 0.0, 0.5233]
A = [-0.2286, -0.0999, 0.2819, 0.0, 0.0, 0.5233]
C = [-0.2975, -0.0527, 0.2872, 0.0, 0.0, 0.5233]
FIXED_SCENE_POSES = {"B": B, "C": C, "A": A}


def _offset_pose(pose: list[float], *, dz: float = 0.0) -> list[float]:
    result = list(pose)
    result[2] = round(result[2] + dz, 4)
    return result


# Fixed B -> C -> A expert trajectory.  All arm targets remain Cartesian;
# hand primitives and timing are interpreted by simulation-web's
# rabo_command_v1 executor.
EXPERT_COMMANDS: tuple[dict[str, Any], ...] = (
    {
        "action_type": "right_arm_move_to",
        "right_moves": [
            {"label": "右臂接近B", "pose": _offset_pose(B, dz=0.10)},
            {"label": "右臂下降至B抓取位", "pose": list(B)},
        ],
    },
    {"action_type": "right_hand_clench"},
    {"action_type": "right_hand_grasp_force"},
    {
        "action_type": "parallel_arm_sequence",
        "right_moves": [
            {"label": "右臂抬起B", "pose": _offset_pose(B, dz=0.12)},
            {"label": "右臂移动B至C上方", "pose": _offset_pose(C, dz=0.12)},
            {"label": "右臂下降至C交接位", "pose": list(C)},
        ],
        "left_moves": [
            {"label": "左臂接近C交接位", "pose": _offset_pose(C, dz=0.12)},
        ],
    },
    {
        "action_type": "left_arm_move_to",
        "left_moves": [
            {"label": "左臂到达C交接位", "pose": list(C)},
        ],
    },
    {"action_type": "left_hand_clench"},
    {"action_type": "left_hand_grasp_force"},
    {"action_type": "wait", "duration_ms": 800},
    {"action_type": "right_hand_open"},
    {"action_type": "wait", "duration_ms": 500},
    {
        "action_type": "parallel_arm_sequence",
        "right_moves": [
            {"label": "右臂退出交接位", "pose": _offset_pose(C, dz=0.15)},
        ],
        "left_moves": [
            {"label": "左臂抬起物体", "pose": _offset_pose(C, dz=0.12)},
            {"label": "左臂移动至A上方", "pose": _offset_pose(A, dz=0.12)},
        ],
    },
    {
        "action_type": "left_arm_move_to",
        "left_moves": [
            {"label": "左臂下降至A放置位", "pose": list(A)},
        ],
    },
    {"action_type": "left_hand_open"},
    {"action_type": "wait", "duration_ms": 500},
    {
        "action_type": "left_arm_move_to",
        "left_moves": [
            {"label": "左臂退出A放置位", "pose": _offset_pose(A, dz=0.12)},
        ],
    },
    {"action_type": "done"},
)


class FixedOraclePolicy:
    name = "fixed_oracle"

    def __init__(self) -> None:
        # Each episode advances independently.  Repeated requests for an
        # already-seen step are idempotent and return the same command.
        self._next_steps: dict[str, int] = {}

    @staticmethod
    def _validate_request(payload: Mapping[str, Any]) -> tuple[str, int]:
        if payload.get("protocol") != RABO_PROTOCOL:
            raise ValueError(f"protocol must be {RABO_PROTOCOL}")

        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")

        episode_id = payload.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")

        step = payload.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")

        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty")

        state = payload.get("state")
        if not isinstance(state, list) or len(state) != 26:
            raise ValueError("state must contain exactly 26 values")

        full_state = payload.get("full_state")
        if not isinstance(full_state, list) or len(full_state) != 36:
            raise ValueError("full_state must contain exactly 36 values")

        images = payload.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("images must be an object")
        camera_names = frozenset(images)
        if camera_names != REQUIRED_CAMERAS:
            expected = ",".join(sorted(REQUIRED_CAMERAS))
            raise ValueError(f"images must contain exactly these cameras: {expected}")
        for camera_name in REQUIRED_CAMERAS:
            camera = images[camera_name]
            if not isinstance(camera, Mapping):
                raise ValueError(f"images.{camera_name} must be an object")
            if camera.get("encoding") != "jpeg_base64":
                raise ValueError(f"images.{camera_name}.encoding must be jpeg_base64")
            data = camera.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError(f"images.{camera_name}.data must be non-empty")

        logger.info(
            "[STATE] request_id=%s episode=%s step=%d state_dim=%d "
            "full_state_dim=%d cameras=%d camera_names=%s",
            request_id,
            episode_id,
            step,
            len(state),
            len(full_state),
            len(camera_names),
            ",".join(sorted(camera_names)),
        )
        return episode_id, step

    async def predict_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        episode_id, step = self._validate_request(payload)

        expected_step = self._next_steps.get(episode_id)
        if expected_step is None:
            if step != 0:
                raise ValueError("a new episode_id must start at step 0")
            self._next_steps[episode_id] = 1
        elif step == expected_step:
            self._next_steps[episode_id] = expected_step + 1
        elif step > expected_step:
            raise ValueError(
                f"episode {episode_id} expected step {expected_step}, received {step}"
            )
        # step < expected_step is an idempotent retry.

        command_index = min(step, len(EXPERT_COMMANDS) - 1)
        command = copy.deepcopy(EXPERT_COMMANDS[command_index])
        logger.info(
            "[ORACLE] episode=%s step=%d action=%s",
            episode_id,
            command_index,
            command["action_type"],
        )
        return {
            "type": "action",
            "protocol": RABO_PROTOCOL,
            "request_id": payload["request_id"],
            "oracle_step": command_index,
            "command": command,
        }

    async def reset(self, episode_id: str | None) -> None:
        if episode_id is None:
            self._next_steps.clear()
        else:
            self._next_steps.pop(episode_id, None)
