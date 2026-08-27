from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class PolicyError(RuntimeError):
    """Base class for errors that can be returned through the bridge."""


class PolicyInputError(PolicyError):
    """The observation cannot be consumed by the selected policy."""


class PolicyNotReadyError(PolicyError):
    """The configured model did not finish loading and warming up."""


@dataclass(frozen=True)
class ImageInput:
    encoding: str
    data: str


@dataclass(frozen=True)
class PolicyRequest:
    request_id: str
    instruction: str | None
    state: Any
    images: dict[str, ImageInput] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PolicyResult:
    action: list[float]
    action_chunk: list[list[float]]
    raw_action_shape: list[int]
    inference_ms: float
    model_name: str
    action_dtype: str = "float32"
    input_mode: str = "request"
    peak_cuda_memory_mb: float | None = None


class BasePolicy(ABC):
    name: str
    output_action_dim: int
    ready: bool = True

    @abstractmethod
    async def predict(self, request: PolicyRequest) -> PolicyResult:
        raise NotImplementedError

    async def warmup(self) -> None:
        return None

    async def reset(self, episode_id: str | None) -> None:
        del episode_id

    @abstractmethod
    def health(self) -> dict[str, Any]:
        raise NotImplementedError


class UnavailablePolicy(BasePolicy):
    """Keeps transport diagnostics available while reporting a failed model load."""

    ready = False

    def __init__(self, name: str, error: str, output_action_dim: int = 0) -> None:
        self.name = name
        self.error = error
        self.output_action_dim = output_action_dim

    async def predict(self, request: PolicyRequest) -> PolicyResult:
        del request
        raise PolicyNotReadyError(self.error)

    def health(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "model_loaded": False,
            "device": None,
            "dtype": None,
            "cuda_memory_allocated_mb": 0.0,
            "error": self.error,
            "output_action_dim": self.output_action_dim,
        }
