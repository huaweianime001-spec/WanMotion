# Motion transfer (CLIP + Wan TI2V)

**Pipeline overview (diagrams + detailed steps):** [`PIPELINE.md`](PIPELINE.md)

Copy of Wan2.2 with a **CLIP motion module** that extracts temporal motion from a driver video and injects it into Wan TI2V via extra cross-attention context tokens.

## Goal

- **Driver video**: `examples/kling_20260512.mp4` (motion from the cat i2v result)
- **Original first frame**: `examples/cat.png`
- **New appearance**: `examples/corgi.png`
- **Output**: i2v video of the corgi performing the driver motion

## Setup

```bash
source activate /home/bro/miniconda3/envs/wan22
cd /home/bro/Wan2.2-motion
pip install open-clip-torch
```

## 1. Train motion module

Trains `MotionClipEncoder` + `MotionContextAdapter` (Wan DiT stays frozen).  
Reconstructs the driver video latents conditioned on `cat.png` + CLIP motion from the driver.

```bash
python train_motion.py \
  --ckpt_dir ../Wan2.2-TI2V-5B \
  --driver_video examples/kling_20260512.mp4 \
  --ref_image examples/cat.png \
  --prompt "the cute cat raises its right paw and licking the paw repeatedly, satisfied and gently" \
  --frame_num 81 \
  --steps 500 \
  --low_vram \
  --output checkpoints/motion_transfer.pt
```

**24GB GPU:** use `--low_vram` (trains at `832*480`, 49 frames). Full `1280*704` + 81 frames OOMs on 24GB.

**If it looks frozen after CLIP loads:** VAE encode can take several minutes. Progress is logged; latents cache to `checkpoints/motion_transfer_latents.pt`.

Use `--no_offload_dit` only if you have 40GB+ VRAM; default keeps DiT on CPU between steps.

Use `--train_clip` to fine-tune temporal layers (more VRAM).

## 2. Inference (corgi + trained motion)

Align motion using **cat.png** (same crop geometry as when the driver was created), appearance from **corgi.png**:

```bash
python generate_motion.py \
  --ckpt_dir ../Wan2.2-TI2V-5B \
  --motion_ckpt checkpoints/motion_transfer.pt \
  --image examples/corgi.png \
  --motion_ref_image examples/cat.png \
  --motion_video examples/kling_20260512.mp4 \
  --prompt "the cute corgi raises its right paw and licking the paw repeatedly, satisfied and gently" \
  --size 1280*704 \
  --frame_num 81 \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --save_file output/corgi_motion.mp4
```

Optional: cache motion once

```bash
python extract_motion.py \
  --motion_video examples/kling_20260512.mp4 \
  --ref_image examples/cat.png \
  --frame_num 81 \
  --output checkpoints/kling_motion.pt

python generate_motion.py ... --motion_cache checkpoints/kling_motion.pt
```

## Architecture

1. **MotionClipEncoder** (`wan/modules/motion_transfer/clip_encoder.py`): frozen OpenCLIP ViT, frame + delta features, temporal transformer → 32 tokens.
2. **MotionContextAdapter**: MLP to Wan `dim=3072`, scaled residual into context.
3. **WanModelWithMotion**: prepends motion tokens to T5 context (same idea as Wan-Animate CLIP tokens).

## Notes

- All tasks are **image-to-video**; first latent frame stays fixed from the input image.
- `frame_num` must be `4n+1` (e.g. 81, 121).
- More training steps usually improve motion fidelity; 500 is a minimal default.
- Base TI2V command (no motion) is unchanged in `inference.md` / `generate.py`.
