"""Supervised Rabo behavior-cloning policy with an Expert shadow fallback."""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import PolicyInputError
from .expert_lookup_policy import EXPERT_FORMAT, RABO_PROTOCOL, REQUIRED_CAMERAS
from .rabo_vla_policy import VLA_ACTION_SPACE, RaboVLAPolicy

logger = logging.getLogger("uvicorn.error")


@dataclass
class _EpisodeProgress:
    cursor: int
    previous_action_id: int


class BCVLAPolicy:
    """Map RGB + 26D proprioception to a trained discrete action id."""

    name = "bc_vla"
    model_name = "RaboBC-VLA-v1"
    ready = True
    output_action_dim = 0
    action_space = VLA_ACTION_SPACE

    def __init__(
        self,
        model_dir: Path,
        expert_program_path: Path,
        *,
        shadow_only: bool = True,
        device: str = "auto",
        guard_max_advance: int = 2,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.expert_program_path = expert_program_path.expanduser().resolve()
        self.shadow_only = bool(shadow_only)
        self.guard_max_advance = int(guard_max_advance)
        if self.guard_max_advance < 0:
            raise ValueError("BC guard_max_advance must be non-negative")

        try:
            import torch
            from PIL import Image, UnidentifiedImageError

            from .bc.dataset import (
                CAMERA_NAMES,
                FULL_STATE_DIM,
                STATE_DIM,
                build_image_transform,
            )
            from .bc.model import BCClassifier, BCModelConfig, sequence_guard
        except ImportError as exc:
            raise RuntimeError(
                "BC policy requires the existing PyTorch/Pillow runtime; configure "
                "PYTHONNOUSERSITE=1 and BC torch site-packages in PYTHONPATH"
            ) from exc

        self._torch = torch
        self._Image = Image
        self._UnidentifiedImageError = UnidentifiedImageError
        self._sequence_guard = sequence_guard
        self._camera_names = tuple(CAMERA_NAMES)
        self._state_dim = int(STATE_DIM)
        self._full_state_dim = int(FULL_STATE_DIM)

        config = self._read_json(self.model_dir / "config.json")
        if config.get("model") != self.model_name:
            raise ValueError(f"BC config model must be {self.model_name}")
        if config.get("uses_request_step_as_input") is not False:
            raise ValueError("BC model must declare uses_request_step_as_input=false")
        if tuple(config.get("cameras") or ()) != self._camera_names:
            raise ValueError("BC model camera order does not match Rabo protocol")
        if config.get("proprio_dim") != self._state_dim:
            raise ValueError("BC model proprio_dim does not match Rabo protocol")

        requested_device = str(device).lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("BC_DEVICE must be auto, cpu, or cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("BC_DEVICE=cuda but CUDA is unavailable")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        checkpoint_path = self.model_dir / "model.pt"
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"BC checkpoint not found: {checkpoint_path}") from exc
        if not isinstance(checkpoint, dict):
            raise TypeError("BC checkpoint root must be a mapping")
        if checkpoint.get("format") != "rabo_bc_checkpoint_v1":
            raise ValueError("BC checkpoint format is unsupported")
        if checkpoint.get("uses_request_step_as_input") is not False:
            raise ValueError("BC checkpoint illegally uses request step as input")
        model_config = BCModelConfig.from_dict(checkpoint["model_config"])
        if model_config.to_dict() != config.get("model_config"):
            raise ValueError("BC checkpoint and config model definitions differ")
        self.num_actions = model_config.num_actions
        self.uses_previous_action_id = model_config.use_previous_action
        self.model = BCClassifier(model_config)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.parameters = self.model.parameter_count

        normalization = checkpoint.get("state_normalization")
        if not isinstance(normalization, dict):
            raise TypeError("BC checkpoint is missing state normalization")
        self._state_mean = torch.tensor(
            normalization.get("mean"), dtype=torch.float32, device=self.device
        )
        self._state_std = torch.tensor(
            normalization.get("std"), dtype=torch.float32, device=self.device
        )
        if self._state_mean.shape != (self._state_dim,) or self._state_std.shape != (
            self._state_dim,
        ):
            raise ValueError("BC checkpoint state normalization has the wrong shape")
        if bool((self._state_std <= 0).any().item()):
            raise ValueError("BC checkpoint state normalization std must be positive")
        self._image_transform = build_image_transform(
            augment=False, image_size=model_config.image_size
        )

        expert = self._read_json(self.expert_program_path)
        if expert.get("format") != EXPERT_FORMAT or expert.get("protocol") != RABO_PROTOCOL:
            raise ValueError("BC Expert action decoder has an invalid program")
        commands = expert.get("commands")
        if not isinstance(commands, list) or expert.get("num_steps") != len(commands):
            raise ValueError("BC Expert action decoder has an invalid command list")
        if len(commands) != self.num_actions:
            raise ValueError(
                f"BC classes ({self.num_actions}) do not match Expert commands ({len(commands)})"
            )

        library = self._read_json(self.model_dir / "action_library.json")
        if library.get("format") != "rabo_bc_action_library_v1":
            raise ValueError("BC action_library.json has an unsupported format")
        actions = library.get("actions")
        if not isinstance(actions, list) or len(actions) != self.num_actions:
            raise ValueError("BC action_library.json has the wrong number of actions")
        validated: list[dict[str, Any]] = []
        for action_id, (item, saved) in enumerate(zip(commands, actions, strict=True)):
            if not isinstance(item, dict) or item.get("step") != action_id:
                raise ValueError(f"Expert action {action_id} is invalid")
            if not isinstance(saved, dict) or saved.get("action_id") != action_id:
                raise ValueError(f"saved BC action {action_id} is invalid")
            if item.get("phase") != saved.get("phase") or item.get("command") != saved.get(
                "command"
            ):
                raise ValueError(
                    f"current Expert action {action_id} differs from trained action library"
                )
            validated.append(copy.deepcopy(item))
        self._commands = tuple(validated)

        self._reference = RaboVLAPolicy(self.expert_program_path)
        self._progress: dict[str, _EpisodeProgress] = {}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"required BC file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"{path} must contain a JSON object")
        return value

    @staticmethod
    def _vector(value: Any, name: str, expected: int) -> list[float]:
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence):
            raise PolicyInputError(f"{name} must be a flat JSON array")
        if len(value) != expected:
            raise PolicyInputError(
                f"{name} must contain exactly {expected} values, got {len(value)}"
            )
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise PolicyInputError(f"{name} contains a non-numeric value") from exc
        if not all(math.isfinite(item) for item in result):
            raise PolicyInputError(f"{name} contains a non-finite value")
        return result

    def _decode_images(self, images: Any):
        if not isinstance(images, Mapping):
            raise PolicyInputError("images must be an object")
        missing = REQUIRED_CAMERAS.difference(images)
        if missing:
            raise PolicyInputError("images is missing cameras: " + ",".join(sorted(missing)))
        tensors = []
        for camera in self._camera_names:
            item = images[camera]
            if not isinstance(item, Mapping):
                raise PolicyInputError(f"images.{camera} must be an object")
            if item.get("encoding") != "jpeg_base64":
                raise PolicyInputError(f"images.{camera}.encoding must be jpeg_base64")
            data = item.get("data")
            if not isinstance(data, str) or not data:
                raise PolicyInputError(f"images.{camera}.data must be non-empty")
            try:
                raw = base64.b64decode(data, validate=True)
                with self._Image.open(io.BytesIO(raw)) as image:
                    rgb = image.convert("RGB")
                    rgb.load()
            except (ValueError, OSError, self._UnidentifiedImageError) as exc:
                raise PolicyInputError(f"images.{camera}.data is not a valid image") from exc
            tensors.append(self._image_transform(rgb))
        return self._torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)

    def _episode_progress(self, episode_id: str) -> _EpisodeProgress:
        progress = self._progress.get(episode_id)
        if progress is None:
            progress = _EpisodeProgress(cursor=0, previous_action_id=self.num_actions)
            self._progress[episode_id] = progress
        return progress

    def _infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Run the network without ever reading observation['step']."""

        if observation.get("protocol") != RABO_PROTOCOL:
            raise PolicyInputError(f"protocol must be {RABO_PROTOCOL}")
        request_id = observation.get("request_id")
        episode_id = observation.get("episode_id")
        instruction = observation.get("instruction")
        if not isinstance(request_id, str) or not request_id.strip():
            raise PolicyInputError("request_id must be a non-empty string")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise PolicyInputError("episode_id must be a non-empty string")
        if not isinstance(instruction, str) or not instruction.strip():
            raise PolicyInputError("instruction must be non-empty")

        state_values = self._vector(observation.get("state"), "state", self._state_dim)
        self._vector(
            observation.get("full_state"), "full_state", self._full_state_dim
        )
        images = self._decode_images(observation.get("images"))
        state = self._torch.tensor(
            state_values, dtype=self._torch.float32, device=self.device
        ).unsqueeze(0)
        state = (state - self._state_mean) / self._state_std
        progress = self._episode_progress(episode_id)
        previous = None
        if self.uses_previous_action_id:
            previous = self._torch.tensor(
                [progress.previous_action_id], dtype=self._torch.long, device=self.device
            )

        with self._torch.inference_mode():
            logits = self.model(images, state, previous)
            probabilities = logits.softmax(dim=1)[0]
        raw_confidence, raw_action = probabilities.max(dim=0)
        raw_action_id = int(raw_action.item())
        action_id, candidates = self._sequence_guard(
            probabilities,
            progress.cursor,
            max_advance=self.guard_max_advance,
        )
        confidence = float(probabilities[action_id].item())
        progress.cursor = action_id
        progress.previous_action_id = action_id
        return {
            "request_id": request_id,
            "episode_id": episode_id,
            "raw_action_id": raw_action_id,
            "raw_confidence": float(raw_confidence.item()),
            "action_id": action_id,
            "confidence": confidence,
            "guard_candidates": list(candidates),
        }

    async def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        prediction = self._infer(observation)
        action_id = int(prediction["action_id"])
        item = self._commands[action_id]

        if self.shadow_only:
            reference = await self._reference.act(observation)
            reference_id = int(reference.get("policy_step", -1))
            match = action_id == reference_id
            logger.info(
                "[BC-SHADOW] predicted=%d reference=%d confidence=%.6f match=%s raw=%d",
                action_id,
                reference_id,
                prediction["confidence"],
                str(match).lower(),
                prediction["raw_action_id"],
            )
            reference.update(
                {
                    "policy": self.name,
                    "model": self.model_name,
                    "backend": "rabo_vla_shadow_reference",
                    "prediction": prediction,
                    "shadow_only": True,
                    "shadow_reference_action_id": reference_id,
                    "shadow_match": match,
                    "implementation": "expert_program_backend_with_bc_shadow",
                    "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
                }
            )
            reference.pop("command", None)
            return reference

        action = RaboVLAPolicy._command_to_action(item["command"])
        logger.info(
            "[BC] episode=%s action_id=%d raw=%d confidence=%.6f phase=%s action=%s",
            prediction["episode_id"],
            action_id,
            prediction["raw_action_id"],
            prediction["confidence"],
            item["phase"],
            action.get("type"),
        )
        return {
            "type": "action",
            "protocol": RABO_PROTOCOL,
            "request_id": prediction["request_id"],
            "episode_id": prediction["episode_id"],
            "policy": self.name,
            "model": self.model_name,
            "backend": self.name,
            "policy_step": action_id,
            "phase": item["phase"],
            "action_space": self.action_space,
            "action": action,
            "prediction": prediction,
            "shadow_only": False,
            "implementation": "trained_behavior_cloning",
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    async def predict_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.act(payload)

    async def reset(self, episode_id: str | None) -> None:
        if episode_id is None:
            self._progress.clear()
        else:
            self._progress[episode_id] = _EpisodeProgress(
                cursor=0, previous_action_id=self.num_actions
            )
        await self._reference.reset(episode_id)

    def health(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "protocol": RABO_PROTOCOL,
            "model": self.model_name,
            "model_family": "vision_language_action",
            "model_loaded": True,
            "ready": True,
            "trained": True,
            "vision_inputs": len(self._camera_names),
            "camera_names": list(self._camera_names),
            "proprio_dim": self._state_dim,
            "full_proprio_dim": self._full_state_dim,
            "language_input": True,
            "language_encoded": False,
            "uses_request_step_as_input": False,
            "uses_previous_action_id": self.uses_previous_action_id,
            "action_space": self.action_space,
            "output_action_dim": self.output_action_dim,
            "num_actions": self.num_actions,
            "parameters": self.parameters,
            "device": str(self.device),
            "dtype": "float32",
            "implementation": "supervised_behavior_cloning",
            "shadow_only": self.shadow_only,
            "sequence_guard_max_advance": self.guard_max_advance,
            "model_path": str(self.model_dir / "model.pt"),
            "controller_loaded": True,
            "controller_steps": len(self._commands),
        }
