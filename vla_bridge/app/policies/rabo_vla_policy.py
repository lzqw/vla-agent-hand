"""VLA-style policy facade for the validated Rabo fixed-scene controller.

The bridge-facing API is intentionally stable: ``act(observation) -> action``.
The v1 implementation keeps the already validated expert-program controller as
its action backend, so the transport and robot execution behavior do not change.
A future learned VLA or server-side IK controller can replace that backend
without changing the Rabo web client contract.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .expert_lookup_policy import ExpertLookupPolicy, RABO_PROTOCOL


logger = logging.getLogger("uvicorn.error")


class RaboVLAPolicy:
    """Hybrid VLA-compatible policy with a stable observation-to-action API."""

    name = "rabo_vla"
    model_name = "RaboVLA-Hybrid-v1"
    ready = True
    output_action_dim = 0
    action_space = "structured_robot_action"

    def __init__(self, expert_path: Path) -> None:
        # The first backend is deliberately the command program that has already
        # completed the B -> C -> A task end-to-end.  Keep it behind the VLA
        # facade so later backends can be swapped without touching the web side.
        self._controller = ExpertLookupPolicy(
            expert_path,
            decision_log_label=None,
        )

    async def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Map a multimodal observation to one executable robot action.

        Observation fields are the same ones sent by the Rabo web client:
        language instruction, three RGB images, 26D proprioception, optional
        36D full proprioception, episode id and policy step.
        """
        started = time.perf_counter()
        response = await self._controller.predict_request(observation)

        policy_step = int(response.pop("oracle_step", observation.get("step", 0)))
        phase = str(response.get("phase", "unknown"))
        command = response.get("command") or {}
        action_type = str(command.get("action_type", "unknown"))

        response.update(
            {
                "policy": self.name,
                "model": self.model_name,
                "backend": self.name,
                "policy_step": policy_step,
                "action_space": self.action_space,
                "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
                # Keep the implementation provenance explicit.  This is a
                # hybrid VLA-compatible controller, not a learned VLA claim.
                "implementation": "expert_program_backend",
            }
        )

        logger.info(
            "[VLA] episode=%s step=%d model=%s phase=%s action=%s",
            response.get("episode_id"),
            policy_step,
            self.model_name,
            phase,
            action_type,
        )
        return response

    async def predict_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Compatibility alias for older bridge dispatchers."""
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
