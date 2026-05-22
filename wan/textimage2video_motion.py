# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan TI2V pipeline with CLIP motion conditioning."""
from __future__ import annotations

import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.motion_transfer.clip_encoder import MotionClipEncoder
from .modules.motion_transfer.model import wrap_wan_model
from .modules.motion_transfer.video_io import align_video_to_image, load_video_frames
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .textimage2video import WanTI2V
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2VMotion(WanTI2V):
    """TI2V with CLIP motion features from a driver video."""

    def __init__(
        self,
        config,
        checkpoint_dir,
        motion_ckpt_path: str | None = None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        num_motion_tokens: int = 32,
        train_motion_encoder: bool = False,
    ):
        self._motion_ckpt_path = motion_ckpt_path
        self._num_motion_tokens = num_motion_tokens
        self._train_motion_encoder = train_motion_encoder
        super().__init__(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_sp=use_sp,
            t5_cpu=t5_cpu,
            init_on_cpu=init_on_cpu,
            convert_model_dtype=convert_model_dtype,
        )

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        model = super()._configure_model(
            model, use_sp, dit_fsdp, shard_fn, convert_model_dtype)
        wrapped = wrap_wan_model(
            model,
            motion_context_len=self._num_motion_tokens,
            motion_dim=512,
        )
        if not self.init_on_cpu and not dit_fsdp:
            wrapped.to(self.device)
        self.motion_encoder = MotionClipEncoder(
            num_motion_tokens=self._num_motion_tokens,
            hidden_dim=512,
            freeze_clip=not self._train_motion_encoder,
        ).to(self.device)
        if motion_ckpt := self._motion_ckpt_path:
            if os.path.isfile(motion_ckpt):
                state = torch.load(motion_ckpt, map_location="cpu")
                if "motion_encoder" in state:
                    self.motion_encoder.load_state_dict(
                        state["motion_encoder"], strict=False)
                if "motion_adapter" in state:
                    wrapped.motion_adapter.load_state_dict(
                        state["motion_adapter"], strict=False)
                logging.info("Loaded motion checkpoint from %s", motion_ckpt)
        return wrapped

    def _load_motion_from_video(
        self,
        motion_video: str,
        ref_image: Image.Image,
        frame_num: int,
        max_area: int = 704 * 1280,
    ) -> torch.Tensor:
        video = load_video_frames(motion_video, frame_num, self.device)
        video, _, _ = align_video_to_image(
            video,
            ref_image,
            max_area=max_area,
            patch_size=self.patch_size,
            vae_stride=self.vae_stride,
        )
        self.motion_encoder.to(self.device)
        return self.motion_encoder(video, frame_batch=16)

    @torch.no_grad()
    def encode_motion(
        self,
        motion_video: str,
        ref_image: Image.Image,
        frame_num: int,
        max_area: int = 704 * 1280,
    ) -> torch.Tensor:
        """Extract motion tokens; optionally cache on disk elsewhere."""
        self.motion_encoder.eval()
        motion = self._load_motion_from_video(
            motion_video, ref_image, frame_num, max_area=max_area)
        # Free CLIP VRAM before the heavy DiT sampling loop
        self.motion_encoder.cpu()
        torch.cuda.empty_cache()
        return motion

    def i2v(
        self,
        input_prompt,
        img,
        max_area=704 * 1280,
        frame_num=121,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        motion_video: str | None = None,
        motion_context: torch.Tensor | None = None,
        ref_image_for_motion: Image.Image | None = None,
        motion_max_area: int | None = None,
    ):
        if motion_context is None and motion_video is not None:
            ref = ref_image_for_motion or img
            motion_context = self.encode_motion(
                motion_video,
                ref,
                frame_num,
                max_area=motion_max_area or max_area,
            )
        elif motion_context is not None:
            motion_context = motion_context.to(self.device)

        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))

        img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        f = frame_num
        seq_len = ((f - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim,
            (f - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img_t])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        with (
            torch.amp.autocast("cuda", dtype=self.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                "context": [context[0]],
                "seq_len": seq_len,
                "motion_context": motion_context,
            }
            arg_null = {
                "context": context_null,
                "seq_len": seq_len,
                "motion_context": None,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]
                timestep = torch.stack(timestep).to(self.device)
                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent
                x0 = [latent]

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            videos = self.vae.decode(x0) if self.rank == 0 else None

        del noise, latent, x0, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return videos[0] if self.rank == 0 else None
