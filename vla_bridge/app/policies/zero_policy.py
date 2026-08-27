from __future__ import annotations

import time

from .base import BasePolicy, PolicyRequest, PolicyResult


class ZeroPolicy(BasePolicy):
    """Transport diagnostic policy; intentionally returns a fixed zero action."""

    name = "zero"

    def __init__(self, action_dim: int) -> None:
        self.output_action_dim = action_dim

    async def predict(self, request: PolicyRequest) -> PolicyResult:
        del request
        started = time.perf_counter()
        action = [0.0] * self.output_action_dim
        return PolicyResult(
            action=action,
            action_chunk=[action.copy()],
            raw_action_shape=[1, self.output_action_dim],
            inference_ms=round((time.perf_counter() - started) * 1000, 3),
            model_name="zero-policy",
            input_mode="zero",
        )

    def health(self) -> dict[str, object]:
        return {
            "policy": self.name,
            "model": "zero-policy",
            "model_loaded": True,
            "device": "cpu",
            "dtype": "float32",
            "cuda_memory_allocated_mb": 0.0,
            "output_action_dim": self.output_action_dim,
        }
