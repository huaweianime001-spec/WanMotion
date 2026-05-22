#!/usr/bin/env python3
"""Pre-extract and cache CLIP motion features from a driver video."""
import argparse
import torch
from PIL import Image

from wan.modules.motion_transfer.clip_encoder import MotionClipEncoder
from wan.modules.motion_transfer.video_io import align_video_to_image, load_video_frames
from wan.configs import MAX_AREA_CONFIGS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_video", required=True)
    p.add_argument("--ref_image", required=True,
                   help="Image used for spatial alignment (e.g. cat.png)")
    p.add_argument("--output", required=True)
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--size", default="1280*704")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}")
    enc = MotionClipEncoder().to(device).eval()
    ref = Image.open(args.ref_image).convert("RGB")
    video = load_video_frames(args.motion_video, args.frame_num, device)
    video, _, _ = align_video_to_image(
        video, ref, MAX_AREA_CONFIGS[args.size], (1, 2, 2), (4, 16, 16))
    with torch.no_grad():
        motion = enc(video)
    torch.save({"motion_context": motion.cpu()}, args.output)
    print("saved", args.output, motion.shape)


if __name__ == "__main__":
    main()
