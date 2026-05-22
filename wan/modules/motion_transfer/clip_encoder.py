# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Extract per-frame and temporal motion features with CLIP."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


class MotionClipEncoder(nn.Module):
    """Encode video motion using a frozen CLIP image encoder.

    Frame embeddings are combined with temporal differences, then compressed
    by a small temporal transformer into a fixed number of motion tokens.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        num_motion_tokens: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.num_motion_tokens = num_motion_tokens
        self.hidden_dim = hidden_dim

        import open_clip

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
            force_quick_gelu=(clip_pretrained == "openai"),
        )
        self.clip_model = clip_model.visual
        self.clip_preprocess = preprocess
        self.clip_dim = clip_model.visual.output_dim

        if freeze_clip:
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad_(False)

        self.frame_proj = nn.Linear(self.clip_dim * 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_motion_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

        # Fallback normalize when open_clip preprocess is PIL-based
        self.tensor_normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] in [-1, 1]. Returns CLIP-ready tensor."""
        x = (frames + 1.0) * 0.5
        x = F.interpolate(
            x.flatten(0, 1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        x = self.tensor_normalize(x)
        return x.view(frames.size(0), frames.size(1), 3, 224, 224)

    def encode_frames(
        self,
        frames: torch.Tensor,
        frame_batch: int = 0,
    ) -> torch.Tensor:
        """Encode frames with CLIP. frames: [B, T, 3, H, W] -> [B, T, D]."""
        b, t, c, h, w = frames.shape
        was_training = self.clip_model.training
        self.clip_model.eval()
        chunks = []
        step = frame_batch if frame_batch > 0 else t
        for start in range(0, t, step):
            end = min(start + step, t)
            x = self._preprocess_frames(frames[:, start:end]).flatten(0, 1)
            chunks.append(self.clip_model(x))
        if was_training:
            self.clip_model.train()
        return torch.cat(chunks, dim=0).view(b, t, -1)

    def forward(
        self,
        frames: torch.Tensor,
        frame_batch: int = 0,
    ) -> torch.Tensor:
        """Return motion tokens [B, num_motion_tokens, hidden_dim]."""
        frame_feats = self.encode_frames(frames, frame_batch=frame_batch)
        b, t, d = frame_feats.shape
        if t > 1:
            delta = frame_feats[:, 1:] - frame_feats[:, :-1]
            delta = F.pad(delta, (0, 0, 0, 1))
        else:
            delta = torch.zeros_like(frame_feats)
        combined = torch.cat([frame_feats, delta], dim=-1)
        combined = self.frame_proj(combined)

        temporal = self.temporal_encoder(combined)
        queries = self.query_tokens.expand(b, -1, -1)
        motion_tokens, _ = self.cross_attn(
            queries, temporal, temporal, need_weights=False)
        return self.out_norm(motion_tokens)
