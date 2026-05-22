# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Video / image loading utilities for motion transfer."""
from __future__ import annotations

import logging

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load RGB image as [1, 3, H, W] in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img).sub_(0.5).div_(0.5)
    return t.unsqueeze(0).to(device)


def load_video_frames(
    path: str,
    frame_num: int,
    device: torch.device,
) -> torch.Tensor:
    """Load video as [1, T, 3, H, W] in [-1, 1], uniformly sampled."""
    try:
        from decord import VideoReader
        vr = VideoReader(path)
        total = len(vr)
        if total <= 0:
            raise ValueError(f"Empty video: {path}")
        indices = np.linspace(0, total - 1, frame_num).astype(np.int64)
        frames = vr.get_batch(indices).asnumpy()
    except Exception as e:
        logger.warning("decord failed (%s), falling back to opencv", e)
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = np.linspace(0, max(total - 1, 0), frame_num).astype(np.int64)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"Could not read video: {path}")
        while len(frames) < frame_num:
            frames.append(frames[-1])
        frames = np.stack(frames[:frame_num])

    tensors = []
    for f in frames:
        img = Image.fromarray(f.astype(np.uint8))
        t = TF.to_tensor(img).sub_(0.5).div_(0.5)
        tensors.append(t)
    video = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    return video


def align_video_to_image(
    video: torch.Tensor,
    ref_image: Image.Image,
    max_area: int,
    patch_size,
    vae_stride,
) -> tuple[torch.Tensor, int, int]:
    """Center-crop / resize video frames to match Wan i2v preprocessing."""
    from ...utils.utils import best_output_size

    ih, iw = ref_image.height, ref_image.width
    dh = patch_size[1] * vae_stride[1]
    dw = patch_size[2] * vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

    scale = max(ow / iw, oh / ih)
    rw, rh = round(iw * scale), round(ih * scale)
    x1 = (rw - ow) // 2
    y1 = (rh - oh) // 2

    out = []
    for i in range(video.shape[1]):
        frame = video[0, i].unsqueeze(0)
        frame = torch.nn.functional.interpolate(
            frame, size=(rh, rw), mode="bilinear", align_corners=False)
        frame = frame[:, :, y1:y1 + oh, x1:x1 + ow]
        out.append(frame.squeeze(0))
    return torch.stack(out, dim=0).unsqueeze(0), ow, oh
