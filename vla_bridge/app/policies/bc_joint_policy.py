"""Behavior-cloned VLA policy for arms plus an Expert hand scheduler."""

from __future__ import annotations

import base64
import io
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .arm_hand_reference import ACTION_SPACE, ARM_DIM, ArmHandReference
from .base import PolicyInputError, PolicyNotReadyError
from .expert_lookup_policy import RABO_PROTOCOL, REQUIRED_CAMERAS


logger = logging.getLogger("uvicorn.error")


class BCJointVLAPolicy:
    name = "bc_joint_vla"
    model_name = "RaboBC-Joint-v1"
    ready = True
    output_action_dim = ARM_DIM
    action_space = ACTION_SPACE

    def __init__(
        self,
        model_dir: Path,
        reference_path: Path,
        hand_events_path: Path,
        *,
        initial_search: int = 250,
        forward_window: int = 80,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        checkpoint_path = self.model_dir / "model.pt"
        config_path = self.model_dir / "config.json"
        if not checkpoint_path.is_file() or not config_path.is_file():
            raise PolicyNotReadyError(
                f"BC joint model is not trained yet: expected {checkpoint_path} and {config_path}"
            )

        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise PolicyNotReadyError("BC joint inference requires torch and Pillow") from exc
        from .bc_joint.model import BCJointModel

        self._torch = torch
        self._Image = Image
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        action_dim = int(checkpoint.get("action_dim", -1))
        proprio_dim = int(checkpoint.get("proprio_dim", -1))
        if action_dim != ARM_DIM or proprio_dim != 36:
            raise PolicyNotReadyError(
                f"BC checkpoint must be 36D proprio -> {ARM_DIM}D arms, "
                f"got {proprio_dim}D -> {action_dim}D"
            )
        self.model = BCJointModel(proprio_dim=proprio_dim, action_dim=action_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.state_mean = checkpoint["state_mean"].float()
        self.state_std = checkpoint["state_std"].float()
        self.target_mean = checkpoint["target_mean"].float()
        self.target_std = checkpoint["target_std"].float()
        image_size = checkpoint.get("image_size", [128, 128])
        self.image_height = int(image_size[0])
        self.image_width = int(image_size[1])
        if tuple(self.state_mean.shape) != (36,) or tuple(self.state_std.shape) != (36,):
            raise PolicyNotReadyError("BC state normalization must be 36D")
        if tuple(self.target_mean.shape) != (ARM_DIM,) or tuple(self.target_std.shape) != (
            ARM_DIM,
        ):
            raise PolicyNotReadyError(f"BC target normalization must be {ARM_DIM}D")
        self.trajectory = ArmHandReference(
            reference_path,
            hand_events_path,
            initial_search=initial_search,
            forward_window=forward_window,
        )

    @staticmethod
    def _vector(value: Any, name: str, expected: int) -> np.ndarray:
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence):
            raise PolicyInputError(f"{name} must be a flat JSON array")
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (expected,) or not np.isfinite(arr).all():
            raise PolicyInputError(f"{name} must contain exactly {expected} finite values")
        return arr

    def _decode_image(self, images: Any, name: str):
        if not isinstance(images, Mapping):
            raise PolicyInputError("images must be an object")
        image = images.get(name)
        if not isinstance(image, Mapping):
            raise PolicyInputError(f"images.{name} must be an object")
        if image.get("encoding") != "jpeg_base64":
            raise PolicyInputError(f"images.{name}.encoding must be jpeg_base64")
        data = image.get("data")
        if not isinstance(data, str) or not data:
            raise PolicyInputError(f"images.{name}.data must be non-empty")
        try:
            raw = base64.b64decode(data, validate=True)
            pil = self._Image.open(io.BytesIO(raw)).convert("RGB")
            pil = pil.resize((self.image_width, self.image_height), self._Image.Resampling.BILINEAR)
            array = np.asarray(pil, dtype=np.float32) / np.float32(255.0)
        except Exception as exc:
            raise PolicyInputError(f"unable to decode images.{name}: {exc}") from exc
        tensor = self._torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

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
        images = observation.get("images")
        missing = REQUIRED_CAMERAS.difference(images or {})
        if missing:
            raise PolicyInputError("images is missing cameras: " + ",".join(sorted(missing)))

        high = self._decode_image(images, "cam_high")
        left = self._decode_image(images, "cam_left_wrist")
        right = self._decode_image(images, "cam_right_wrist")
        state = self._torch.from_numpy(current.copy()).unsqueeze(0).float()
        normalized = (state - self.state_mean.unsqueeze(0)) / self.state_std.unsqueeze(0)

        with self._torch.inference_mode():
            pred_norm, progress_tensor = self.model(high, left, right, normalized)
            target = pred_norm * self.target_std.unsqueeze(0) + self.target_mean.unsqueeze(0)
        action = target.squeeze(0).cpu().numpy().astype(np.float32)
        progress = float(progress_tensor.item())
        if not np.isfinite(action).all() or not np.isfinite(progress):
            raise RuntimeError("BC model produced non-finite output")
        if action.shape != (ARM_DIM,):
            raise RuntimeError(f"BC model produced action shape {action.shape}, expected ({ARM_DIM},)")

        aligned = self.trajectory.align(episode_id, current[:ARM_DIM])
        hand_type = (
            aligned.hand_command.get("action_type") if aligned.hand_command is not None else None
        )

        logger.info(
            "[BC-JOINT] episode=%s ref=%d progress=%.4f hand=%s done=%s action_norm=%.4f",
            episode_id,
            aligned.reference_index,
            progress,
            hand_type,
            aligned.done,
            float(np.linalg.norm(action - current[:ARM_DIM])),
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
            "hand_command": aligned.hand_command,
            "done": aligned.done,
            "prediction": {
                "progress": round(progress, 6),
                "reference_index": aligned.reference_index,
                "reference_frames": self.trajectory.num_frames,
                "match_distance": round(aligned.match_distance, 7),
                "hand_event_id": aligned.hand_event_id,
                "uses_request_step_as_input": False,
            },
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "implementation": "single_episode_behavior_cloning",
        }

    async def reset(self, episode_id: str | None) -> None:
        self.trajectory.reset(episode_id)

    def health(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "protocol": RABO_PROTOCOL,
            "model": self.model_name,
            "model_family": "behavior_cloning",
            "model_loaded": True,
            "ready": True,
            "vision_inputs": 3,
            "proprio_dim": 26,
            "full_proprio_dim": 36,
            "language_input": True,
            "action_space": self.action_space,
            "output_action_dim": self.output_action_dim,
            "uses_request_step_as_input": False,
            "implementation": "single_episode_behavior_cloning",
            "model_dir": str(self.model_dir),
            "reference_loaded": True,
            "reference_frames": self.trajectory.num_frames,
            "reference_file": str(self.trajectory.reference_path),
            "hand_events_loaded": True,
            "hand_event_count": len(self.trajectory.events),
            "hand_events_file": str(self.trajectory.hand_events_path),
            "device": "cpu",
            "dtype": "float32",
            "cuda_memory_allocated_mb": 0.0,
        }
