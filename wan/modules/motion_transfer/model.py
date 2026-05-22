# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""WanModel extension that injects CLIP motion tokens into cross-attention."""
from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from ..model import WanModel, sinusoidal_embedding_1d
from .adapter import MotionContextAdapter


class WanModelWithMotion(WanModel):
    """Wan diffusion backbone with optional CLIP motion context tokens."""

    def __init__(self, motion_context_len: int = 32, motion_dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.motion_context_len = motion_context_len
        self.motion_adapter = MotionContextAdapter(
            motion_dim=motion_dim,
            wan_dim=self.dim,
        )

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        motion_context=None,
    ):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t_flat).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        context_lens = None
        text_ctx = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if motion_context is not None:
            motion_ctx = self.motion_adapter(motion_context)
            context = torch.cat([motion_ctx, text_ctx], dim=1)
        else:
            context = text_ctx

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        use_ckpt = self.training and motion_context is not None

        def _block(block, x, kwargs):
            return block(x, **kwargs)

        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(
                    _block, block, x, kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)

        x = self.head(x, e)
        return self.unpatchify(x, grid_sizes)


def wrap_wan_model(
    wan_model: WanModel,
    motion_context_len: int = 32,
    motion_dim: int = 512,
) -> WanModelWithMotion:
    """Wrap a loaded WanModel, copying weights and adding motion adapter."""
    cfg = dict(wan_model.config)
    defaults = dict(
        patch_size=(1, 2, 2),
        text_len=512,
        text_dim=4096,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
    )
    for key, val in defaults.items():
        cfg.setdefault(key, val)
    wrapped = WanModelWithMotion(
        motion_context_len=motion_context_len,
        motion_dim=motion_dim,
        **cfg,
    )
    wrapped.load_state_dict(wan_model.state_dict(), strict=False)
    return wrapped
