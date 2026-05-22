# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Project CLIP motion tokens into Wan cross-attention context space."""
from __future__ import annotations

import torch
import torch.nn as nn


class MotionContextAdapter(nn.Module):
    """Map motion tokens to Wan transformer context dimension."""

    def __init__(
        self,
        motion_dim: int,
        wan_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        dim = motion_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(dim, wan_dim),
                nn.GELU(approximate="tanh"),
            ])
            dim = wan_dim
        layers.append(nn.Linear(dim, wan_dim))
        self.mlp = nn.Sequential(*layers)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, motion_tokens: torch.Tensor) -> torch.Tensor:
        """motion_tokens [B, L, motion_dim] -> [B, L, wan_dim]."""
        return self.mlp(motion_tokens) * self.scale
