"""Small shared-CNN behavior-cloning classifier for Rabo observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class BCModelConfig:
    num_actions: int
    num_cameras: int = 3
    proprio_dim: int = 26
    image_size: int = 224
    image_embedding_dim: int = 96
    proprio_embedding_dim: int = 64
    fusion_hidden_dim: int = 256
    use_previous_action: bool = False
    previous_action_embedding_dim: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BCModelConfig:
        return cls(**value)


class SharedImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(images))


class BCClassifier(nn.Module):
    """Classify an observation into an Expert trajectory action id.

    There is deliberately no ``step`` argument.  The only inputs are RGB,
    26D proprioception and, when explicitly configured, the previous emitted
    action id.
    """

    def __init__(self, config: BCModelConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = SharedImageEncoder(config.image_embedding_dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.proprio_dim, config.proprio_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.proprio_embedding_dim, config.proprio_embedding_dim),
            nn.LayerNorm(config.proprio_embedding_dim),
            nn.ReLU(inplace=True),
        )

        previous_dim = 0
        self.previous_action_embedding: nn.Embedding | None = None
        if config.use_previous_action:
            self.previous_action_embedding = nn.Embedding(
                config.num_actions + 1,
                config.previous_action_embedding_dim,
            )
            previous_dim = config.previous_action_embedding_dim

        fusion_input_dim = (
            config.num_cameras * config.image_embedding_dim
            + config.proprio_embedding_dim
            + previous_dim
        )
        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(config.fusion_hidden_dim, config.num_actions),
        )

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        previous_action_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape [batch,cameras,3,height,width]")
        batch, cameras, channels, height, width = images.shape
        if cameras != self.config.num_cameras or channels != 3:
            raise ValueError(
                f"expected {self.config.num_cameras} RGB cameras, got {tuple(images.shape)}"
            )
        if state.shape != (batch, self.config.proprio_dim):
            raise ValueError(
                f"state must have shape [{batch},{self.config.proprio_dim}]"
            )

        flat_images = images.reshape(batch * cameras, channels, height, width)
        image_features = self.image_encoder(flat_images).reshape(batch, -1)
        features = [image_features, self.proprio_encoder(state)]
        if self.previous_action_embedding is not None:
            if previous_action_id is None or previous_action_id.shape != (batch,):
                raise ValueError("previous_action_id must have shape [batch]")
            features.append(self.previous_action_embedding(previous_action_id))
        return self.head(torch.cat(features, dim=1))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def sequence_guard(
    probabilities: torch.Tensor,
    cursor: int,
    *,
    max_advance: int = 2,
) -> tuple[int, tuple[int, ...]]:
    """Choose the most likely non-regressing id near the server-side cursor."""

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty 1D tensor")
    if max_advance < 0:
        raise ValueError("max_advance must be non-negative")
    last = probabilities.numel() - 1
    start = max(0, min(int(cursor), last))
    candidates = tuple(range(start, min(last, start + max_advance) + 1))
    candidate_tensor = torch.tensor(candidates, device=probabilities.device)
    local_index = int(torch.argmax(probabilities[candidate_tensor]).item())
    return candidates[local_index], candidates
