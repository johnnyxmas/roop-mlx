"""FLUX.1 Kontext instruction editing, in MLX.

Kontext is a rectified-flow DiT that edits by *reference*: the source image
is VAE-encoded, packed into tokens and appended to the noisy canvas tokens,
tagged with RoPE axis 0 = 1 so the transformer can tell reference from
canvas.  The prompt therefore describes the change, not the whole scene.

Memory note: the transformer is ~24 GB in bf16 and T5-XXL another ~9.5 GB.
The text encoders are loaded, used and released per generation, and the
transformer is not built until the first prompt has been encoded -- so the
first run peaks at roughly the transformer alone.  Later runs re-encode
against a resident transformer and do peak at the sum (~34 GB), which is
why `unload()` exists.
"""

import gc
import json
import os
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image

from roop.diffusion import loader, tokenizer
from roop.diffusion.nn.clip_text import CLIPTextModel
from roop.diffusion.nn.flux import FluxTransformer2DModel
from roop.diffusion.nn.t5 import T5EncoderModel
from roop.diffusion.nn.vae import AutoencoderKL
from roop.diffusion.registry import ModelSpec
from roop.diffusion.sampler import FlowMatchScheduler

ProgressFn = Optional[Callable[[float, str], None]]

MAX_T5_TOKENS = 512

# Canvas + reference + prompt tokens. 1536x1536 against a bucketed reference
# lands near 14k; beyond ~20k the attention matrices stop fitting in a single
# Metal buffer, so refuse rather than crash in the allocator.
MAX_SEQUENCE_TOKENS = 20000


# -- latent packing ---------------------------------------------------------
def pack(latent: mx.array) -> mx.array:
    """(b, h, w, c) -> (b, h*w/4, c*4), grouping each 2x2 patch into a token."""
    b, h, w, c = latent.shape
    x = latent.reshape(b, h // 2, 2, w // 2, 2, c)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(tokens: mx.array, h: int, w: int, c: int) -> mx.array:
    """Inverse of `pack`, back to (b, h, w, c) latents."""
    b = tokens.shape[0]
    x = tokens.reshape(b, h // 2, w // 2, c, 2, 2)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(b, h, w, c)


def latent_ids(h: int, w: int, index: int = 0) -> mx.array:
    """RoPE position ids for a packed h/2 x w/2 token grid."""
    gh, gw = h // 2, w // 2
    ids = np.zeros((gh, gw, 3), dtype=np.float32)
    ids[..., 0] = index
    ids[..., 1] = np.arange(gh)[:, None]
    ids[..., 2] = np.arange(gw)[None, :]
    return mx.array(ids.reshape(gh * gw, 3))


def _t5_tokenize(root: str, prompts: List[str]) -> mx.array:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(root, "tokenizer_2", "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_T5_TOKENS)
    tok.enable_padding(length=MAX_T5_TOKENS, pad_id=0, pad_token="<pad>")
    encoded = tok.encode_batch(prompts)
    return mx.array([e.ids for e in encoded], dtype=mx.int32)


# Kontext was trained with the reference image snapped to one of these
# buckets -- all ~1024x1024 in area, varying only in aspect.  Matching them
# matters twice over: it is what the model expects, and it bounds the
# reference to ~4096 tokens no matter how large a photo the user drops in.
# Attention is quadratic in sequence length, so an unbounded reference is
# the difference between a 3-second edit and a 30 GB allocation failure.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]


def reference_size(width: int, height: int) -> tuple:
    """Bucket a reference image to the nearest aspect Kontext was trained on."""
    aspect = width / max(height, 1)
    return min(PREFERRED_KONTEXT_RESOLUTIONS,
               key=lambda wh: abs(wh[0] / wh[1] - aspect))


def _to_nhwc(image: Image.Image, width: int, height: int) -> mx.array:
    img = image.convert("RGB").resize((width, height), Image.LANCZOS)
    return mx.array(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)[None]


def _to_pil(x: mx.array) -> List[Image.Image]:
    arr = np.array(mx.clip((x + 1.0) * 127.5, 0, 255).astype(mx.uint8))
    return [Image.fromarray(a) for a in arr]


def _scheduler_config(root: str) -> dict:
    path = os.path.join(root, "scheduler", "scheduler_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class FluxKontextPipeline:
    def __init__(self, spec: ModelSpec, root: str, progress: ProgressFn = None):
        self.spec = spec
        self.root = root

        def report(f, m):
            if progress:
                progress(f, m)

        report(0.3, "Loading VAE…")
        vae_cfg = loader.load_config(root, "vae")
        self.vae = AutoencoderKL(vae_cfg)
        loader.load_component(self.vae, root, "vae", dtype=mx.float32)
        self.vae_scale = 2 ** (len(vae_cfg["block_out_channels"]) - 1)
        self.latent_channels = vae_cfg.get("latent_channels", 16)

        # The transformer is ~24 GB and T5-XXL another ~9.5 GB.  It is loaded
        # lazily, *after* the first prompt is encoded and the text encoders
        # have been released, so the two never have to be resident together
        # on the first run.
        self.tf_cfg = loader.load_config(root, "transformer")
        self.transformer = None

        self.sched_cfg = _scheduler_config(root)
        report(1.0, "Model ready")

    def _ensure_transformer(self, progress: ProgressFn = None) -> None:
        if self.transformer is not None:
            return
        if progress:
            progress(0.0, "Loading transformer (~24 GB, first run only)…")
        transformer = FluxTransformer2DModel(self.tf_cfg)
        loader.load_component(transformer, self.root, "transformer",
                              dtype=mx.bfloat16)
        self.transformer = transformer

    # -- conditioning -----------------------------------------------------
    def _encode_prompt(self, prompt: str, batch: int,
                       progress: ProgressFn) -> Tuple[mx.array, mx.array]:
        """Run both text encoders, then release them before sampling."""
        if progress:
            progress(0.0, "Encoding prompt (CLIP)…")
        clip_cfg = loader.load_config(self.root, "text_encoder")
        clip = CLIPTextModel(clip_cfg)
        loader.load_component(clip, self.root, "text_encoder",
                              dtype=mx.bfloat16)
        tokens = tokenizer.tokenize([prompt] * batch, pad_id=tokenizer.EOT)
        _, _, pooled = clip(tokens)
        mx.eval(pooled)
        del clip
        gc.collect()
        mx.clear_cache()

        if progress:
            progress(0.0, "Encoding prompt (T5-XXL)…")
        t5_cfg = loader.load_config(self.root, "text_encoder_2")
        t5 = T5EncoderModel(t5_cfg)
        loader.load_component(t5, self.root, "text_encoder_2",
                              dtype=mx.bfloat16)
        embeds = t5(_t5_tokenize(self.root, [prompt] * batch))
        mx.eval(embeds)
        del t5
        gc.collect()
        mx.clear_cache()

        return embeds, pooled

    # -- sampling ---------------------------------------------------------
    def generate(self, prompt: str, steps: int = 28, guidance: float = 2.5,
                 seed: int = -1, width: int = 1024, height: int = 1024,
                 batch: int = 1, image: Optional[Image.Image] = None,
                 progress: ProgressFn = None,
                 **_ignored) -> List[Image.Image]:

        if image is None:
            raise ValueError(
                f"{self.spec.label} is an editing model -- it needs an input "
                "image to work from."
            )

        # Latent grids must be even so 2x2 packing divides cleanly.
        width = max(int(width) // 16 * 16, 256)
        height = max(int(height) // 16 * 16, 256)
        if seed is None or seed < 0:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        key = mx.random.key(int(seed))

        embeds, pooled = self._encode_prompt(prompt, batch, progress)
        embeds = embeds.astype(mx.bfloat16)
        pooled = pooled.astype(mx.bfloat16)
        self._ensure_transformer(progress)

        lat_h = height // self.vae_scale
        lat_w = width // self.vae_scale
        c = self.latent_channels

        # Reference image: snapped to a trained bucket, never to whatever
        # resolution happened to come off the user's camera.
        ref_w, ref_h = reference_size(image.width, image.height)
        ref_latent = self.vae.encode(_to_nhwc(image, ref_w, ref_h))
        ref_lat_h, ref_lat_w = ref_latent.shape[1], ref_latent.shape[2]
        ref_tokens = pack(ref_latent).astype(mx.bfloat16)
        ref_tokens = mx.broadcast_to(
            ref_tokens, (batch,) + ref_tokens.shape[1:]
        )

        key, noise_key = mx.random.split(key)
        noise = mx.random.normal((batch, lat_h, lat_w, c), key=noise_key)
        latents = pack(noise).astype(mx.bfloat16)

        img_ids = mx.concatenate(
            [latent_ids(lat_h, lat_w, 0),
             latent_ids(ref_lat_h, ref_lat_w, 1)], axis=0
        )
        txt_ids = mx.zeros((embeds.shape[1], 3))

        # Attention cost is quadratic in this, so fail with something the user
        # can act on rather than a raw allocator error thousands of lines deep.
        seq_len = img_ids.shape[0] + embeds.shape[1]
        if seq_len > MAX_SEQUENCE_TOKENS:
            raise ValueError(
                f"{width}x{height} needs {seq_len} tokens, over the "
                f"{MAX_SEQUENCE_TOKENS} this pipeline allows. Reduce the "
                "output width/height."
            )

        scheduler = FlowMatchScheduler(self.sched_cfg)
        scheduler.set_timesteps(steps, seq_len=latents.shape[1])

        guidance_vec = mx.full((batch,), float(guidance))
        n_target = latents.shape[1]

        for i in range(steps):
            model_in = mx.concatenate([latents, ref_tokens], axis=1)
            t = mx.full((batch,), float(scheduler.sigmas[i]))
            out = self.transformer(model_in, embeds, pooled, t,
                                   img_ids, txt_ids, guidance_vec)
            out = out[:, :n_target]
            latents = scheduler.step(latents, out.astype(latents.dtype), i)
            mx.eval(latents)
            if progress:
                progress((i + 1) / steps, f"Step {i + 1}/{steps}")

        decoded = self.vae.decode(
            unpack(latents.astype(mx.float32), lat_h, lat_w, c)
        )
        mx.eval(decoded)
        return _to_pil(decoded)
