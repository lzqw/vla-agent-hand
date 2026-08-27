"""VLA-style policy facade for the validated Rabo fixed-scene controller.

The bridge-facing API is intentionally stable: ``act(observation) -> action``.
The v2 implementation exposes a real top-level VLA ``action`` object while
keeping the already validated expert-program controller behind the policy.
A future learned VLA or server-side IK backend can replace the controller
without changing the Rabo web client contract.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .expert_lookup_policy import ExpertLookupPolicy, RABO_PROTOCOL


logger = logging.getLogger("uvicorn.error")
VLA_ACTION_SPACE = "rabo_vla_action_v1"


class RaboVLAPolicy:
    """Hybrid VLA-compatible policy with a stable observation-to-action API."""

    name = "rabo_vla"
    model_name = "RaboVLA-Hybrid-v2"
    ready = True
    output_action_dim = 0
    action_space = VLA_ACTION_SPACE

    def __init__(self, expert_path: Path) -> None:
        self._controller = ExpertLookupPolicy(
            expert_path,
            decision_log_label=None,
        )

    @staticmethod
    def _moves(command: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
        raw = command.get(key) or []
        return [copy.deepcopy(item) for item in raw]

    @classmethod
    def _command_to_action(cls, command: Mapping[str, Any]) -> dict[str, Any]:
        """Translate the validated controller command into a VLA action object."""
        action_type = str(command.get("action_type", ""))

        if action_type == "right_arm_move_to":
            return {
                "type": "pose_trajectory",
                "effector": "right_arm",
                "trajectory": cls._moves(command, "right_moves"),
            }
        if action_type == "left_arm_move_to":
            return {
                "type": "pose_trajectory",
                "effector": "left_arm",
                "trajectory": cls._moves(command, "left_moves"),
            }
        if action_type == "parallel_arm_sequence":
            return {
                "type": "bimanual_pose_trajectory",
                "left_trajectory": cls._moves(command, "left_moves"),
                "right_trajectory": cls._moves(command, "right_moves"),
            }
        if action_type in {"right_hand_clench", "left_hand_clench"}:
            return {
                "type": "hand_control",
                "effector": "right_hand" if action_type.startswith("right") else "left_hand",
                "mode": "clench",
                "values": copy.deepcopy(command.get("clench")),
            }
        if action_type in {"right_hand_grasp_force", "left_hand_grasp_force"}:
            action: dict[str, Any] = {
                "type": "hand_control",
                "effector": "right_hand" if action_type.startswith("right") else "left_hand",
                "mode": "grasp_force",
                "strength": float(command.get("strength", 0.0)),
            }
            if command.get("fingers") is not None:
                action["fingers"] = [int(v) for v in command["fingers"]]
            return action
        if action_type == "wait":
            return {"type": "wait", "duration_s": float(command.get("duration_s", 0.0))}
        if action_type == "done":
            return {"type": "done"}
        raise ValueError(f"unsupported controller action_type: {action_type!r}")

    async def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Map multimodal observation to one externally visible VLA action."""
        started = time.perf_counter()
        response = await self._controller.predict_request(observation)

        policy_step = int(response.pop("oracle_step", observation.get("step", 0)))
        phase = str(response.get("phase", "unknown"))
        command = response.pop("command", None)
        if not isinstance(command, Mapping):
            raise ValueError("controller did not return a command object")

        action = self._command_to_action(command)
        response.update(
            {
                "policy": self.name,
                "model": self.model_name,
                "backend": self.name,
                "policy_step": policy_step,
                "action_space": self.action_space,
                "action": action,
                "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "implementation": "expert_program_backend",
            }
        )

        logger.info(
            "[VLA] episode=%s step=%d model=%s phase=%s action=%s",
            response.get("episode_id"),
            policy_step,
            self.model_name,
            phase,
            action.get("type"),
        )
        return response

    async def predict_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.act(payload)

    async def reset(self, episode_id: str | None) -> None:
        await self._controller.reset(episode_id)

    def health(self) -> dict[str, Any]:
        backend = self._controller.health()
        return {
            "policy": self.name,
            "protocol": RABO_PROTOCOL,
            "model": self.model_name,
            "model_family": "vision_language_action",
            "model_loaded": True,
            "ready": True,
            "vision_inputs": 3,
            "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "proprio_dim": 26,
            "full_proprio_dim": 36,
            "language_input": True,
            "action_space": self.action_space,
            "output_action_dim": self.output_action_dim,
            "device": "cpu",
            "dtype": None,
            "cuda_memory_allocated_mb": 0.0,
            "implementation": "expert_program_backend",
            "controller_loaded": True,
            "controller_steps": backend.get("expert_steps"),
            "controller_file": backend.get("expert_file"),
        }
