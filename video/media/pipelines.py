"""GPU pipelines: Z-Image-Turbo (text->image) and Wan 2.2 TI2V-5B (text/image->video).

Only one pipeline lives on the GPU at a time; the orchestrator decides which.
"""

from __future__ import annotations

import gc
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("pipelines")

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
# "model" = enable_model_cpu_offload (safe on 24 GB), "none" = keep all on GPU
IMAGE_OFFLOAD = os.getenv("IMAGE_OFFLOAD", "model")
VIDEO_OFFLOAD = os.getenv("VIDEO_OFFLOAD", "model")

VIDEO_RES = {"480p": (480, 832), "720p": (704, 1280)}  # (h, w) landscape
WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)

ProgressCb = Callable[[float], None]


def _seed(seed: int) -> int:
    return seed if seed is not None and seed >= 0 else random.randint(0, 2**31 - 1)


def _step_callback(total: int, progress: ProgressCb, start: float = 0.0, span: float = 0.95):
    def cb(pipe, step: int, timestep, callback_kwargs):
        progress(start + span * (step + 1) / max(total, 1))
        return callback_kwargs

    return cb


def _place(pipe, offload: str):
    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


class PipelineManager:
    def __init__(self) -> None:
        self.kind: str | None = None
        self.pipe = None
        self.i2v = None
        self._video_mode: str | None = None

    # ---------------------------------------------------------- lifecycle
    def unload(self) -> None:
        if self.kind is None:
            return
        log.info("unloading %s pipeline", self.kind)
        self.pipe = None
        self.i2v = None
        self.kind = None
        self._video_mode = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load(self, kind: str) -> None:
        if kind == self.kind:
            return
        self.unload()
        t0 = time.time()
        if kind == "image":
            from diffusers import ZImagePipeline

            pipe = ZImagePipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.bfloat16)
            self.pipe = _place(pipe, IMAGE_OFFLOAD)
        elif kind == "video":
            from diffusers import AutoencoderKLWan, WanPipeline

            vae = AutoencoderKLWan.from_pretrained(VIDEO_MODEL, subfolder="vae", torch_dtype=torch.float32)
            pipe = WanPipeline.from_pretrained(VIDEO_MODEL, vae=vae, torch_dtype=torch.bfloat16)
            pipe.vae.enable_tiling()
            self.pipe = pipe
            self._use_video_mode("t2v")
        else:
            raise ValueError(f"unknown pipeline kind {kind}")
        self.kind = kind
        log.info("loaded %s pipeline in %.1fs", kind, time.time() - t0)

    def _use_video_mode(self, mode: str):
        """Switch offload hooks between the shared T2V and I2V pipelines."""
        if mode == self._video_mode:
            return self.i2v if mode == "i2v" else self.pipe
        if mode == "i2v" and self.i2v is None:
            from diffusers import WanImageToVideoPipeline

            c = self.pipe.components
            self.i2v = WanImageToVideoPipeline(
                tokenizer=c["tokenizer"],
                text_encoder=c["text_encoder"],
                vae=c["vae"],
                scheduler=c["scheduler"],
                transformer=c["transformer"],
                transformer_2=c.get("transformer_2"),
                boundary_ratio=self.pipe.config.get("boundary_ratio"),
                expand_timesteps=True,
            )
        target = self.i2v if mode == "i2v" else self.pipe
        other = self.pipe if mode == "i2v" else self.i2v
        if other is not None and VIDEO_OFFLOAD != "none":
            other.remove_all_hooks()
        _place(target, VIDEO_OFFLOAD)
        self._video_mode = mode
        return target

    # ----------------------------------------------------------- generate
    def generate_image(self, req: dict, out_path: Path, progress: ProgressCb) -> dict:
        self.load("image")
        seed = _seed(req.get("seed", -1))
        width = int(req.get("width", 1024)) // 16 * 16
        height = int(req.get("height", 1024)) // 16 * 16
        steps = int(req.get("steps", 9))
        t0 = time.time()
        with torch.inference_mode():
            image = self.pipe(
                prompt=req["prompt"],
                negative_prompt=req.get("negative_prompt") or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=float(req.get("guidance", 0.0)),
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=_step_callback(steps, progress),
            ).images[0]
        image.save(out_path)
        progress(1.0)
        return {"path": str(out_path), "seed": seed, "width": width, "height": height, "seconds": round(time.time() - t0, 2)}

    def generate_video(self, req: dict, out_path: Path, progress: ProgressCb) -> dict:
        from diffusers.utils import export_to_video

        self.load("video")
        seed = _seed(req.get("seed", -1))
        h, w = VIDEO_RES[req.get("resolution", "480p")]
        frames = int(req.get("num_frames", 81))
        frames = (frames - 1) // 4 * 4 + 1  # Wan needs 4k+1 frames
        steps = int(req.get("steps", 30))
        kwargs = dict(
            prompt=req["prompt"],
            negative_prompt=req.get("negative_prompt") or WAN_NEGATIVE,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=float(req.get("guidance", 5.0)),
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=_step_callback(steps, progress, 0.0, 0.9),
        )
        t0 = time.time()
        image_path = req.get("image_path")
        if image_path:
            pipe = self._use_video_mode("i2v")
            image = Image.open(image_path).convert("RGB")
            # keep the input aspect ratio at the requested pixel budget
            area = h * w
            mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            ar = image.height / image.width
            h = max(mod, round(np.sqrt(area * ar)) // mod * mod)
            w = max(mod, round(np.sqrt(area / ar)) // mod * mod)
            kwargs["image"] = image.resize((w, h), Image.Resampling.LANCZOS)
        else:
            pipe = self._use_video_mode("t2v")
        with torch.inference_mode():
            video = pipe(height=h, width=w, **kwargs).frames[0]
        progress(0.95)
        fps = int(req.get("fps", 24))
        export_to_video(video, str(out_path), fps=fps)
        progress(1.0)
        return {
            "path": str(out_path),
            "seed": seed,
            "width": w,
            "height": h,
            "frames": frames,
            "fps": fps,
            "seconds": round(time.time() - t0, 2),
        }
