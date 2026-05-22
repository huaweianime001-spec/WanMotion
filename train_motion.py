#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Train CLIP motion adapter for Wan TI2V motion transfer."""
import argparse
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm

from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.modules.motion_transfer.video_io import (
    align_video_to_image,
    load_video_frames,
)
from wan.textimage2video_motion import WanTI2VMotion
from wan.utils.utils import masks_like

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def log(msg, *args):
    if args:
        logger.info(msg, *args)
    else:
        logger.info(msg)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description="Train Wan motion transfer module")
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--driver_video", type=str, required=True)
    p.add_argument("--ref_image", type=str, required=True,
                   help="First-frame reference used for original i2v (e.g. cat.png)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output", type=str, default="checkpoints/motion_transfer.pt")
    p.add_argument("--latent_cache", type=str, default=None,
                   help="Cache file for VAE latents (.pt). Reused if exists.")
    p.add_argument("--size", type=str, default="1280*704")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--train_clip", action="store_true",
                   help="Also fine-tune CLIP temporal layers (uses more VRAM)")
    p.add_argument("--offload_dit", action="store_true", default=True,
                   help="Keep DiT on CPU between steps to save VRAM (default: on)")
    p.add_argument("--no_offload_dit", action="store_false", dest="offload_dit",
                   help="Keep full DiT on GPU (faster steps, needs more VRAM)")
    p.add_argument("--clip_frame_batch", type=int, default=16,
                   help="CLIP frames per micro-batch during motion encoding")
    p.add_argument(
        "--low_vram",
        action="store_true",
        help="Train at 832*480 and max 49 frames to fit ~24GB GPUs",
    )
    p.add_argument(
        "--reencode_latents",
        action="store_true",
        help="Ignore latent cache and re-run VAE encode",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")
    cfg = WAN_CONFIGS["ti2v-5B"]
    if args.low_vram:
        args.size = "832*480"
        args.frame_num = min(args.frame_num, 49)
        args.reencode_latents = True
        log("low_vram: using size=%s frame_num=%d (re-encoding latents)",
            args.size, args.frame_num)
    max_area = MAX_AREA_CONFIGS[args.size]

    log("Loading Wan TI2V + motion modules (DiT on CPU to save VRAM)...")
    pipe = WanTI2VMotion(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
        t5_cpu=True,
        convert_model_dtype=True,
        init_on_cpu=True,
        train_motion_encoder=args.train_clip,
    )
    pipe.motion_encoder.to(device)
    pipe.vae.model.to(device)
    if not args.offload_dit:
        pipe.model.to(device)

    log("Loading driver video: %s", args.driver_video)
    ref_img = Image.open(args.ref_image).convert("RGB")
    video_bthw = load_video_frames(args.driver_video, args.frame_num, device)
    video_bthw, ow, oh = align_video_to_image(
        video_bthw,
        ref_img,
        max_area=max_area,
        patch_size=pipe.patch_size,
        vae_stride=pipe.vae_stride,
    )
    video_cthw = video_bthw.squeeze(0).permute(1, 0, 2, 3).contiguous()
    log(
        "Video shape [C,T,H,W]=%s (VAE encode can take 5-15 min at 720p+; not frozen)",
        list(video_cthw.shape),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    cache_path = args.latent_cache
    if cache_path is None:
        cache_path = args.output.replace(".pt", "_latents.pt")
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.isfile(cache_path) and not args.reencode_latents:
        log("Loading cached latents from %s", cache_path)
        cache = torch.load(cache_path, map_location=device)
        target_z = cache["target_z"]
        cond_z = cache["cond_z"]
        motion_ctx = cache.get("motion_ctx")
        if motion_ctx is not None:
            motion_ctx = motion_ctx.to(device)
    else:
        log("Encoding full video with VAE (please wait)...")
        t0 = time.time()
        with torch.no_grad():
            target_z = pipe.vae.encode([video_cthw])[0]
        log("VAE target latent done in %.1fs, shape=%s",
            time.time() - t0, list(target_z.shape))
        torch.cuda.empty_cache()

        log("Encoding first frame with VAE...")
        t0 = time.time()
        with torch.no_grad():
            cond_z = pipe.vae.encode([video_cthw[:, :1]])[0]
        log("VAE cond latent done in %.1fs", time.time() - t0)

        log("Encoding motion with CLIP...")
        t0 = time.time()
        with torch.no_grad():
            motion_ctx = pipe.motion_encoder(
                video_bthw, frame_batch=args.clip_frame_batch)
        log("CLIP motion done in %.1fs, shape=%s",
            time.time() - t0, list(motion_ctx.shape))

        torch.save(
            {
                "target_z": target_z.cpu(),
                "cond_z": cond_z.cpu(),
                "motion_ctx": motion_ctx.cpu(),
                "ow": ow,
                "oh": oh,
            },
            cache_path,
        )
        log("Saved latent cache to %s", cache_path)
        target_z = target_z.to(device)
        cond_z = cond_z.to(device)
        motion_ctx = motion_ctx.to(device)

    pipe.vae.model.cpu()
    if not args.train_clip:
        del video_bthw, video_cthw
    torch.cuda.empty_cache()
    log("VAE offloaded from GPU for training")

    log("Encoding prompt with T5 on CPU (can take 1-3 min)...")
    t0 = time.time()
    with torch.no_grad():
        context = pipe.text_encoder([args.prompt], torch.device("cpu"))
        context = [t.to(device) for t in context]
    log("T5 prompt encoding done in %.1fs", time.time() - t0)

    pipe.model.train()
    for p in pipe.model.parameters():
        p.requires_grad_(False)
    pipe.model.motion_adapter.requires_grad_(True)
    pipe.motion_encoder.requires_grad_(args.train_clip)
    for p in pipe.motion_encoder.clip_model.parameters():
        p.requires_grad_(False)

    trainable = [p for p in pipe.motion_encoder.parameters() if p.requires_grad]
    trainable += list(pipe.model.motion_adapter.parameters())
    optim = AdamW(trainable, lr=args.lr)
    log("Trainable parameters: %d", sum(p.numel() for p in trainable))

    _, t_lat, h_lat, w_lat = target_z.shape
    seq_len = (t_lat * h_lat * w_lat) // (
        pipe.patch_size[1] * pipe.patch_size[2])
    log("seq_len=%d (latent %dx%dx%d), starting training (%d steps)...",
        seq_len, t_lat, h_lat, w_lat, args.steps)
    log("First DiT forward is slow (~1-3 min/step with offload). Watch tqdm.")

    mask1, mask2 = masks_like([target_z], zero=True)

    if not args.train_clip:
        motion_ctx_fixed = motion_ctx.detach()

    for step in tqdm(range(1, args.steps + 1), desc="train"):
        optim.zero_grad(set_to_none=True)

        if args.train_clip:
            motion_ctx_step = pipe.motion_encoder(
                video_bthw, frame_batch=args.clip_frame_batch)
        else:
            motion_ctx_step = motion_ctx_fixed

        noise = torch.randn_like(target_z)
        t_scalar = torch.randint(
            1, pipe.num_train_timesteps, (1,), device=device)
        sigma = t_scalar.float() / pipe.num_train_timesteps
        noisy = (1.0 - sigma) * target_z + sigma * noise
        noisy = (1.0 - mask2[0]) * cond_z + mask2[0] * noisy

        # Match wan/textimage2video.py i2v: scalar t broadcast via mask, then pad to seq_len
        t_val = t_scalar.to(dtype=target_z.dtype)
        temp_ts = (mask2[0][0][:, ::2, ::2] * t_val).flatten()
        if temp_ts.size(0) < seq_len:
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_val,
            ])
        timestep_b = temp_ts.unsqueeze(0)

        if args.offload_dit:
            pipe.model.to(device)
            torch.cuda.empty_cache()
        try:
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype):
                pred = pipe.model(
                    [noisy],
                    t=timestep_b,
                    context=[context[0]],
                    seq_len=seq_len,
                    motion_context=motion_ctx_step,
                )[0]
            target = noise - target_z
            loss = F.mse_loss(pred.float(), target.float())
            loss.backward()
            optim.step()
        finally:
            if args.offload_dit:
                pipe.model.cpu()
                torch.cuda.empty_cache()

        if step == 1 or step % 20 == 0:
            log("step %d loss %.6f", step, loss.item())
        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "motion_encoder": pipe.motion_encoder.state_dict(),
                    "motion_adapter": pipe.model.motion_adapter.state_dict(),
                    "step": step,
                    "loss": loss.item(),
                    "prompt": args.prompt,
                    "ref_image": args.ref_image,
                    "driver_video": args.driver_video,
                    "frame_num": args.frame_num,
                },
                args.output,
            )
            log("saved %s", args.output)

    log("Training finished: %s", args.output)


if __name__ == "__main__":
    main()
