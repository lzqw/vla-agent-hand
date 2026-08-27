from __future__ import annotations

import torch
from torch import nn


class SharedImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCJointModel(nn.Module):
    """Small multimodal BC network: 3 RGB + 36D proprio -> next 36D joints."""

    def __init__(
        self,
        *,
        proprio_dim: int = 36,
        action_dim: int = 36,
        image_embedding_dim: int = 96,
        proprio_embedding_dim: int = 96,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = SharedImageEncoder(image_embedding_dim)
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proprio_embedding_dim),
            nn.ReLU(inplace=True),
        )
        fusion_dim = image_embedding_dim * 3 + proprio_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.progress_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        cam_high: torch.Tensor,
        cam_left_wrist: torch.Tensor,
        cam_right_wrist: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.encoder(cam_high)
        h1 = self.encoder(cam_left_wrist)
        h2 = self.encoder(cam_right_wrist)
        hp = self.proprio(proprio)
        fused = self.fusion(torch.cat([h0, h1, h2, hp], dim=-1))
        return self.action_head(fused), self.progress_head(fused).squeeze(-1)
