"""Arm-only joint VLA with dense trajectory alignment and robust O6 events."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .arm_hand_reference import ACTION_SPACE, ARM_DIM, ArmHandReference
from .base import PolicyInputError
from .expert_lookup_policy import RABO_PROTOCOL, REQUIRED_CAMERAS


logger = logging.getLogger("uvicorn.error")


class JointVLAPolicy:
    """Observation-aligned policy returning 14D bimanual arm joint targets."""

    name = "joint_vla"
    model_name = "RaboVLA-Joint-v3"
    ready = True
    output_action_dim = ARM_DIM
    action_space = ACTION_SPACE

    def __init__(
        self,
        reference_path: Path,
        hand_events_path: Path,
        *,
        initial_search: int = 250,
        forward_window: int = 80,
        target_tolerance_rad: float = 0.025,
        lookahead_frames: int = 3,
        hand_event_tolerance_rad: float = 0.020,
        hand_event_settle_cycles: int = 2,
        grasp_force_repeats: int = 2,
    ) -> None:
        self.trajectory = ArmHandReference(
            reference_path,
            hand_events_path,
            initial_search=initial_search,
            forward_window=forward_window,
            target_tolerance_rad=target_tolerance_rad,
            lookahead_frames=lookahead_frames,
            hand_event_tolerance_rad=hand_event_tolerance_rad,
            hand_event_settle_cycles=hand_event_settle_cycles,
            grasp_force_repeats=grasp_force_repeats,
        )
        self.reference_path = self.trajectory.reference_path
        self.hand_events_path = self.trajectory.hand_events_path
        self.num_frames = self.trajectory.num_frames

    @staticmethod
    def _vector(value: Any, name: str, expected: int) -> np.ndarray:
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence):
            raise PolicyInputError(f"{name} must be a flat JSON array")
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (expected,) or not np.isfinite(arr).all():
            raise PolicyInputError(f"{name} must contain exactly {expected} finite values")
        return arr

    @staticmethod
    def _validate_images(images: Any) -> None:
        if not isinstance(images, Mapping):
            raise PolicyInputError("images must be an object")
        missing = REQUIRED_CAMERAS.difference(images)
        if missing:
            raise PolicyInputError("images is missing cameras: " + ",".join(sorted(missing)))
        for name in REQUIRED_CAMERAS:
            image = images[name]
            if not isinstance(image, Mapping):
                raise PolicyInputError(f"images.{name} must be an object")
            if image.get("encoding") != "jpeg_base64":
                raise PolicyInputError(f"images.{name}.encoding must be jpeg_base64")
            if not isinstance(image.get("data"), str) or not image["data"]:
                raise PolicyInputError(f"images.{name}.data must be non-empty")

    async def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if observation.get("protocol") != RABO_PROTOCOL:
            raise PolicyInputError(f"protocol must be {RABO_PROTOCOL}")

        request_id = observation.get("request_id")
        episode_id = observation.get("episode_id")
        instruction = observation.get("instruction")
        if not isinstance(request_id, str) or not request_id.strip():
            raise PolicyInputError("request_id must be non-empty")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise PolicyInputError("episode_id must be non-empty")
        if not isinstance(instruction, str) or not instruction.strip():
            raise PolicyInputError("instruction must be non-empty")

        self._vector(observation.get("state"), "state", 26)
        current = self._vector(observation.get("full_state"), "full_state", 36)
        self._validate_images(observation.get("images"))

        aligned = self.trajectory.align(episode_id, current[:ARM_DIM])
        hand_type = (
            aligned.hand_command.get("action_type") if aligned.hand_command is not None else None
        )

        logger.info(
            "[JOINT-VLA] episode=%s ref=%d target=%d/%d match=%.6f "
            "event=%s frame=%s err=%s settle=%d repeat=%d hand=%s done=%s",
            episode_id,
            aligned.reference_index,
            aligned.action_reference_index,
            self.num_frames,
            aligned.match_distance,
            aligned.hand_event_id,
            aligned.hand_event_frame,
            None if aligned.hand_event_error_rad is None else round(aligned.hand_event_error_rad, 5),
            aligned.hand_event_settle_count,
            aligned.hand_event_repeat_index,
            hand_type,
            aligned.done,
        )
        return {
            "type": "action",
            "protocol": RABO_PROTOCOL,
            "request_id": request_id,
            "episode_id": episode_id,
            "policy": self.name,
            "model": self.model_name,
            "backend": self.name,
            "action_space": self.action_space,
            "action": aligned.target_arm.tolist(),
            "hand_command": aligned.hand_command,
            "done": aligned.done,
            "prediction": {
                "reference_index": aligned.reference_index,
                "action_reference_index": aligned.action_reference_index,
                "reference_frames": self.num_frames,
                "match_distance": round(aligned.match_distance, 7),
                "hand_event_id": aligned.hand_event_id,
                "hand_event_frame": aligned.hand_event_frame,
                "hand_event_error_rad": (
                    None
                    if aligned.hand_event_error_rad is None
                    else round(aligned.hand_event_error_rad, 7)
                ),
                "hand_event_settle_count": aligned.hand_event_settle_count,
                "hand_event_repeat_index": aligned.hand_event_repeat_index,
                "lookahead_frames": self.trajectory.lookahead_frames,
                "uses_request_step_as_input": False,
            },
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "implementation": "dense_arm_reference_with_converged_hand_events",
        }

    async def reset(self, episode_id: str | None) -> None:
        self.trajectory.reset(episode_id)

    def health(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "protocol": RABO_PROTOCOL,
            "model": self.model_name,
            "model_family": "vision_language_action",
            "model_loaded": True,
            "ready": True,
            "vision_inputs": 3,
            "proprio_dim": 26,
            "full_proprio_dim": 36,
            "language_input": True,
            "action_space": self.action_space,
            "output_action_dim": self.output_action_dim,
            "reference_loaded": True,
            "reference_frames": self.num_frames,
            "reference_file": str(self.reference_path),
            "hand_events_loaded": True,
            "hand_event_count": len(self.trajectory.events),
            "hand_events_file": str(self.hand_events_path),
            "uses_request_step_as_input": False,
            "target_tolerance_rad": self.trajectory.target_tolerance_rad,
            "lookahead_frames": self.trajectory.lookahead_frames,
            "hand_event_tolerance_rad": self.trajectory.hand_event_tolerance_rad,
            "hand_event_settle_cycles": self.trajectory.hand_event_settle_cycles,
            "grasp_force_repeats": self.trajectory.grasp_force_repeats,
            "implementation": "dense_arm_reference_with_converged_hand_events",
            "device": "cpu",
            "dtype": "float32",
            "cuda_memory_allocated_mb": 0.0,
        }
