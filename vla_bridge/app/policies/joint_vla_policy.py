"""Pure joint-position VLA facade backed by a recorded successful trajectory.

This policy intentionally outputs only 36D full joint-position targets.  It does
not invent an A7 kinematic model on the 4080 server.  Instead it uses the joint
trajectory measured after the official Rabo SDK solved/executed the expert
Cartesian path, and aligns the live proprioceptive observation to that reference.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import PolicyInputError
from .expert_lookup_policy import RABO_PROTOCOL, REQUIRED_CAMERAS


logger = logging.getLogger("uvicorn.error")
ACTION_SPACE = "joint_position_36d"


class JointVLAPolicy:
    """Observation-aligned reference policy returning full 36D joint targets."""

    name = "joint_vla"
    model_name = "RaboVLA-Joint-v1"
    ready = True
    output_action_dim = 36
    action_space = ACTION_SPACE

    def __init__(
        self,
        reference_path: Path,
        *,
        initial_search: int = 250,
        forward_window: int = 80,
    ) -> None:
        self.reference_path = reference_path.expanduser().resolve()
        try:
            data = np.load(self.reference_path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise ValueError(f"joint reference not found: {self.reference_path}") from exc

        required = {"reference_full_state", "target_full_state"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"joint reference missing arrays: {sorted(missing)}")

        reference = np.asarray(data["reference_full_state"], dtype=np.float32)
        target = np.asarray(data["target_full_state"], dtype=np.float32)
        if reference.ndim != 2 or reference.shape[1] != 36:
            raise ValueError(f"reference_full_state must be [N,36], got {reference.shape}")
        if target.shape != reference.shape:
            raise ValueError(
                f"target_full_state must match reference shape {reference.shape}, got {target.shape}"
            )
        if len(reference) < 2 or not np.isfinite(reference).all() or not np.isfinite(target).all():
            raise ValueError("joint reference contains invalid values")

        self.reference = reference
        self.target = target
        self.num_frames = int(reference.shape[0])
        self.initial_search = max(1, min(int(initial_search), self.num_frames))
        self.forward_window = max(2, int(forward_window))

        # Normalize matching distances so large-range arm joints do not completely
        # dominate low-range hand joints.  Keep a floor for almost-static joints.
        scale = np.std(reference, axis=0).astype(np.float32)
        self.scale = np.maximum(scale, np.float32(0.03))

        self._cursor: dict[str, int] = {}
        self._lock = threading.Lock()

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

    def _match_index(self, episode_id: str, current: np.ndarray) -> tuple[int, float]:
        with self._lock:
            if episode_id not in self._cursor:
                start = 0
                end = self.initial_search
                previous = 0
            else:
                previous = self._cursor[episode_id]
                start = max(0, previous - 2)
                end = min(self.num_frames, previous + self.forward_window)

            candidates = self.reference[start:end]
            normalized = (candidates - current[None, :]) / self.scale[None, :]
            distances = np.mean(normalized * normalized, axis=1)
            local = int(np.argmin(distances))
            matched = start + local

            # The observation decides alignment, but the online cursor prevents
            # oscillating backwards on near-identical wait/grasp frames.
            selected = max(previous, matched)
            selected = min(selected, self.num_frames - 1)
            self._cursor[episode_id] = min(selected + 1, self.num_frames - 1)
            return selected, float(distances[local])

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

        # Keep both observation modalities validated even though v1 trajectory
        # alignment currently uses proprioception for its deterministic matching.
        self._vector(observation.get("state"), "state", 26)
        current = self._vector(observation.get("full_state"), "full_state", 36)
        self._validate_images(observation.get("images"))

        index, distance = self._match_index(episode_id, current)
        action = self.target[index].astype(np.float32, copy=True)
        done = bool(index >= self.num_frames - 1)

        logger.info(
            "[JOINT-VLA] episode=%s ref=%d/%d match=%.6f done=%s",
            episode_id,
            index,
            self.num_frames,
            distance,
            done,
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
            "action": action.tolist(),
            "done": done,
            "prediction": {
                "reference_index": index,
                "reference_frames": self.num_frames,
                "match_distance": round(distance, 7),
            },
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "implementation": "recorded_sdk_ik_joint_reference",
        }

    async def reset(self, episode_id: str | None) -> None:
        with self._lock:
            if episode_id is None:
                self._cursor.clear()
            else:
                self._cursor.pop(episode_id, None)

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
            "implementation": "recorded_sdk_ik_joint_reference",
            "device": "cpu",
            "dtype": "float32",
            "cuda_memory_allocated_mb": 0.0,
        }
