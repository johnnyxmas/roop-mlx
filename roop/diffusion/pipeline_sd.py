"""SD 1.5 / SDXL text-to-image and image-to-image, in MLX.

One class covers both: SDXL differs only in having a second text encoder,
a pooled + size-conditioning vector fed to the UNet, and a 1024px default.
"""

import json
import os
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from roop.diffusion import loader, tokenizer
from roop.diffusion.nn.clip_text import CLIPTextModel
from roop.diffusion.nn.unet import UNet2DConditionModel
from roop.diffusion.nn.vae import AutoencoderKL
from roop.diffusion.registry import ARCH_SDXL, ModelSpec
from roop.diffusion.sampler import SDScheduler


ProgressFn = Optional[Callable[[float, str], None]]


def _scheduler_config(root: str) -> dict:
    path = os.path.join(root, "scheduler", "scheduler_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_nhwc(image: Image.Image, width: int, height: int) -> mx.array:
    img = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return mx.array(arr)[None]


def _to_pil(x: mx.array) -> List[Image.Image]:
    arr = np.array(mx.clip((x + 1.0) * 127.5, 0, 255).astype(mx.uint8))
    return [Image.fromarray(a) for a in arr]


class SDPipeline:
    """A loaded SD 1.5 or SDXL checkpoint, ready to sample from."""

    def __init__(self, spec: ModelSpec, root: str, progress: ProgressFn = None):
        self.spec = spec
        self.root = root
        self.is_xl = spec.arch == ARCH_SDXL

        def report(f, m):
            if progress:
                progress(f, m)

        report(0.05, "Loading text encoder…")
        te_cfg = loader.load_config(root, "text_encoder")
        self.text_encoder = CLIPTextModel(te_cfg)
        loader.load_component(self.text_encoder, root, "text_encoder",
                              spec.variant, dtype=mx.float16)

        self.text_encoder_2 = None
        if self.is_xl:
            report(0.2, "Loading second text encoder…")
            te2_cfg = loader.load_config(root, "text_encoder_2")
            self.text_encoder_2 = CLIPTextModel(te2_cfg, with_projection=True)
            loader.load_component(self.text_encoder_2, root, "text_encoder_2",
                                  spec.variant, dtype=mx.float16)

        report(0.4, "Loading UNet…")
        unet_cfg = loader.load_config(root, "unet")
        self.unet = UNet2DConditionModel(unet_cfg)
        loader.load_component(self.unet, root, "unet", spec.variant,
                              dtype=mx.float16)

        report(0.85, "Loading VAE…")
        vae_cfg = loader.load_config(root, "vae")
        self.vae = AutoencoderKL(vae_cfg)
        # fp32: the SDXL VAE overflows fp16 on decode, which shows up as
        # black or magenta output rather than an error.
        loader.load_component(self.vae, root, "vae", spec.variant,
                              dtype=mx.float32)

        self.sched_cfg = _scheduler_config(root)
        self.vae_scale = 2 ** (len(vae_cfg["block_out_channels"]) - 1)
        self.latent_channels = unet_cfg.get("in_channels", 4)
        report(1.0, "Model ready")

    # -- conditioning -----------------------------------------------------
    def _encode_prompt(self, prompts: List[str], clip_skip: int):
        tokens = tokenizer.tokenize(prompts, pad_id=tokenizer.EOT)
        if not self.is_xl:
            embeds, _ = self.text_encoder.encode(tokens, clip_skip=clip_skip)
            return embeds, None

        emb1, _ = self.text_encoder.encode(tokens, penultimate=True)
        tokens2 = tokenizer.tokenize(prompts, pad_id=0)
        emb2, pooled = self.text_encoder_2.encode(tokens2, penultimate=True)
        return mx.concatenate([emb1, emb2], axis=-1), pooled

    # -- sampling ---------------------------------------------------------
    def generate(self, prompt: str, negative_prompt: str = "",
                 steps: int = 25, guidance: float = 7.5, seed: int = -1,
                 width: int = 512, height: int = 512, strength: float = 0.65,
                 sampler: str = "Euler", clip_skip: int = 1, batch: int = 1,
                 image: Optional[Image.Image] = None,
                 progress: ProgressFn = None) -> List[Image.Image]:

        width = int(width) // 8 * 8
        height = int(height) // 8 * 8
        if seed is None or seed < 0:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        key = mx.random.key(int(seed))

        scheduler = SDScheduler(self.sched_cfg, sampler)
        scheduler.set_timesteps(steps)

        do_cfg = guidance > 1.0
        cond, cond_pooled = self._encode_prompt([prompt] * batch, clip_skip)
        if do_cfg:
            uncond, uncond_pooled = self._encode_prompt(
                [negative_prompt or ""] * batch, clip_skip
            )
            context = mx.concatenate([uncond, cond], axis=0)
        else:
            context = cond
        context = context.astype(mx.float16)

        added = None
        if self.is_xl:
            time_ids = mx.array(
                [[height, width, 0, 0, height, width]] * batch,
                dtype=mx.float32,
            )
            if do_cfg:
                time_ids = mx.concatenate([time_ids, time_ids], axis=0)
                pooled = mx.concatenate([uncond_pooled, cond_pooled], axis=0)
            else:
                pooled = cond_pooled
            added = {"text_embeds": pooled.astype(mx.float16),
                     "time_ids": time_ids}

        lat_h, lat_w = height // self.vae_scale, width // self.vae_scale
        lat_c = self.latent_channels
        key, noise_key = mx.random.split(key)
        noise = mx.random.normal((batch, lat_h, lat_w, lat_c), key=noise_key)

        if image is not None:
            start = scheduler.start_index(steps, strength)
            init = self.vae.encode(_to_nhwc(image, width, height)
                                   .astype(mx.float32))
            init = mx.broadcast_to(init, (batch, lat_h, lat_w, lat_c))
            latents = scheduler.add_noise(init.astype(mx.float32),
                                          noise, start)
        else:
            start = 0
            latents = noise * scheduler.init_sigma
        latents = latents.astype(mx.float16)

        total = len(scheduler.timesteps) - start
        for n, i in enumerate(range(start, len(scheduler.timesteps))):
            model_in = scheduler.scale_input(latents, i)
            if do_cfg:
                model_in = mx.concatenate([model_in, model_in], axis=0)
            t = mx.array([scheduler.timesteps[i]] * model_in.shape[0])

            out = self.unet(model_in, t, context, added)
            if do_cfg:
                uncond_out, cond_out = mx.split(out, 2, axis=0)
                out = uncond_out + guidance * (cond_out - uncond_out)

            key, step_key = mx.random.split(key)
            latents = scheduler.step(latents, out, i, step_key)
            mx.eval(latents)
            if progress:
                progress((n + 1) / total, f"Step {n + 1}/{total}")

        decoded = self.vae.decode(latents.astype(mx.float32))
        mx.eval(decoded)
        return _to_pil(decoded)
