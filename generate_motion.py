#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Image-to-video with transferred motion from a driver video."""
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import random
import torch
from PIL import Image

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.textimage2video_motion import WanTI2VMotion
from wan.utils.utils import save_video, str2bool


def parse_args():
    p = argparse.ArgumentParser(
        description="Wan TI2V with CLIP motion transfer (image-to-video)")
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--motion_ckpt", type=str, required=True)
    p.add_argument("--image", type=str, required=True,
                   help="Target appearance image (e.g. corgi.png)")
    p.add_argument("--motion_video", type=str, required=True,
                   help="Driver video for motion (e.g. kling_20260512.mp4)")
    p.add_argument("--motion_ref_image", type=str, default=None,
                   help="Reference for motion crop alignment (default: same as --image; "
                        "use cat.png if driver was generated from cat)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--size", type=str, default="1280*704")
    p.add_argument("--frame_num", type=int, default=None)
    p.add_argument("--motion_cache", type=str, default=None,
                   help="Optional .pt cache of precomputed motion tokens")
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--offload_model", type=str2bool, default=None,
                   help="Offload DiT to CPU each step (saves VRAM, very slow). "
                        "Default: False if --keep_gpu, else True.")
    p.add_argument(
        "--keep_gpu",
        action="store_true",
        help="Keep DiT on GPU for whole sampling (much faster, needs ~20GB+ VRAM)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Preset: keep_gpu, 40 steps, 49 frames, 832*480",
    )
    p.add_argument("--convert_model_dtype", action="store_true", default=False)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--sample_guide_scale", type=float, default=None)
    p.add_argument("--sample_solver", type=str, default="unipc")
    p.add_argument("--base_seed", type=int, default=-1)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    cfg = WAN_CONFIGS["ti2v-5B"]
    if args.fast:
        args.keep_gpu = True
        args.size = "832*480"
        if args.frame_num is None:
            args.frame_num = 81
        if args.sample_steps is None:
            args.sample_steps = 40

    # Default 81 matches common i2v usage; cfg default is 121 (slower).
    frame_num = args.frame_num if args.frame_num is not None else 81
    sampling_steps = args.sample_steps or cfg.sample_steps
    offload_model = args.offload_model
    if offload_model is None:
        offload_model = not args.keep_gpu
    init_on_cpu = not args.keep_gpu
    shift = args.sample_shift or cfg.sample_shift
    guide_scale = args.sample_guide_scale or cfg.sample_guide_scale
    seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    img = Image.open(args.image).convert("RGB")
    motion_ref = Image.open(args.motion_ref_image or args.image).convert("RGB")

    if offload_model:
        logging.warning(
            "offload_model=True moves the 5B DiT CPU<->GPU every step. "
            "On 24GB this often means hours per video. Use --keep_gpu or --fast "
            "if you have enough VRAM."
        )

    pipe = WanTI2VMotion(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        motion_ckpt_path=args.motion_ckpt,
        device_id=args.device,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        init_on_cpu=init_on_cpu,
    )

    if args.motion_cache and os.path.isfile(args.motion_cache):
        logging.info("Loading cached motion from %s", args.motion_cache)
        motion_context = torch.load(args.motion_cache, map_location="cpu")
        if isinstance(motion_context, dict):
            motion_context = motion_context.get("motion_context", motion_context)
        motion_context = motion_context.to(pipe.device)
    else:
        motion_context = pipe.encode_motion(
            args.motion_video,
            motion_ref,
            frame_num,
            max_area=MAX_AREA_CONFIGS[args.size],
        )
        if args.motion_cache:
            torch.save({"motion_context": motion_context.cpu()}, args.motion_cache)
            logging.info("Cached motion to %s", args.motion_cache)

    logging.info("Generating i2v with motion transfer ...")
    video = pipe.i2v(
        input_prompt=args.prompt,
        img=img,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=frame_num,
        shift=shift,
        sample_solver=args.sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
        motion_context=motion_context,
        ref_image_for_motion=motion_ref,
        motion_max_area=MAX_AREA_CONFIGS[args.size],
    )

    if args.save_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_file = f"motion_i2v_{ts}.mp4"
    logging.info("Saving to %s", args.save_file)
    save_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
