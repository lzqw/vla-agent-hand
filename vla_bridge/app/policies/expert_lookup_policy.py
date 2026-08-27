"""Validated fixed-scene Expert lookup policy for rabo_command_v1."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import PolicyInputError


logger = logging.getLogger("uvicorn.error")

EXPERT_FORMAT = "rabo_expert_program_v1"
RABO_PROTOCOL = "rabo_command_v1"
REQUIRED_CAMERAS = frozenset(
    {"cam_high", "cam_left_wrist", "cam_right_wrist"}
)


class ExpertLookupPolicy:
    """Load a command program once and return its item for the requested step."""

    name = "expert_lookup"
    ready = True
    output_action_dim = 0

    def __init__(self, expert_path: Path) -> None:
        self.expert_path = expert_path.expanduser().resolve()
        try:
            raw = json.loads(self.expert_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Expert file not found: {self.expert_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expert file is not valid JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError("Expert file root must be a JSON object")
        if raw.get("format") != EXPERT_FORMAT:
            raise ValueError(f"Expert format must be {EXPERT_FORMAT}")
        if raw.get("protocol") != RABO_PROTOCOL:
            raise ValueError(f"Expert protocol must be {RABO_PROTOCOL}")

        commands = raw.get("commands")
        if not isinstance(commands, list):
            raise ValueError("Expert commands must be a JSON array")
        if raw.get("num_steps") != len(commands):
            raise ValueError("Expert num_steps must equal len(commands)")

        for index, item in enumerate(commands):
            if not isinstance(item, dict):
                raise ValueError(f"Expert commands[{index}] must be an object")
            if item.get("step") != index:
                raise ValueError(f"Expert commands[{index}].step must equal {index}")
            if not isinstance(item.get("phase"), str) or not item["phase"]:
                raise ValueError(f"Expert commands[{index}].phase must be non-empty")
            command = item.get("command")
            if not isinstance(command, dict):
                raise ValueError(f"Expert commands[{index}].command must be an object")
            action_type = command.get("action_type")
            if not isinstance(action_type, str) or not action_type:
                raise ValueError(
                    f"Expert commands[{index}].command.action_type must be non-empty"
                )

        self._commands: tuple[dict[str, Any], ...] = tuple(commands)

    @staticmethod
    def _dimension(value: Any, name: str, expected: int) -> int:
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence):
            raise PolicyInputError(f"{name} must be a flat JSON array")
        dimension = len(value)
        if dimension != expected:
            raise PolicyInputError(
                f"{name} must contain exactly {expected} values, got {dimension}"
            )
        return dimension

    @staticmethod
    def _validate_images(images: Any) -> list[str]:
        if not isinstance(images, Mapping):
            raise PolicyInputError("images must be an object")
        missing = REQUIRED_CAMERAS.difference(images)
        if missing:
            raise PolicyInputError(
                "images is missing cameras: " + ",".join(sorted(missing))
            )
        for name in REQUIRED_CAMERAS:
            image = images[name]
            if not isinstance(image, Mapping):
                raise PolicyInputError(f"images.{name} must be an object")
            if image.get("encoding") != "jpeg_base64":
                raise PolicyInputError(f"images.{name}.encoding must be jpeg_base64")
            data = image.get("data")
            if not isinstance(data, str) or not data:
                raise PolicyInputError(f"images.{name}.data must be non-empty")
        return sorted(REQUIRED_CAMERAS)

    async def predict_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("protocol") != RABO_PROTOCOL:
            raise PolicyInputError(f"protocol must be {RABO_PROTOCOL}")

        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise PolicyInputError("request_id must be a non-empty string")
        episode_id = payload.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise PolicyInputError("episode_id must be a non-empty string")
        step = payload.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise PolicyInputError("step must be a non-negative integer")
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise PolicyInputError("instruction must be non-empty")

        state_dim = self._dimension(payload.get("state"), "state", 26)
        full_state_dim = self._dimension(
            payload.get("full_state"), "full_state", 36
        )
        camera_names = self._validate_images(payload.get("images"))
        logger.info(
            "[STATE] episode=%s step=%d state_dim=%d full_state_dim=%d cameras=%d",
            episode_id,
            step,
            state_dim,
            full_state_dim,
            len(camera_names),
        )

        if step < len(self._commands):
            item = self._commands[step]
            oracle_step = int(item["step"])
            phase = str(item["phase"])
            command = copy.deepcopy(item["command"])
        else:
            oracle_step = step
            phase = "completed"
            command = {"action_type": "done"}

        logger.info(
            "[EXPERT] step=%d phase=%s action=%s",
            oracle_step,
            phase,
            command["action_type"],
        )
        return {
            "type": "action",
            "protocol": RABO_PROTOCOL,
            "request_id": request_id,
            "episode_id": episode_id,
            "oracle_step": oracle_step,
            "phase": phase,
            "command": command,
            "backend": self.name,
        }

    async def reset(self, episode_id: str | None) -> None:
        # Lookup is a pure function of payload.step, so no server-side episode
        # cursor needs to be reset.
        del episode_id

    def health(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "protocol": RABO_PROTOCOL,
            "expert_loaded": True,
            "expert_steps": len(self._commands),
            "expert_file": str(self.expert_path),
            "model": "fixed-scene-expert",
            "model_loaded": True,
            "device": "cpu",
            "dtype": None,
            "cuda_memory_allocated_mb": 0.0,
            "output_action_dim": self.output_action_dim,
        }
