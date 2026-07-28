"""Entry point the UI talks to: fetch, load, cache and run a diffusion model.

Exactly one pipeline stays resident at a time.  These checkpoints are 2-24
GB, so switching models frees the previous one before loading the next
rather than letting two coexist.
"""

import gc
import json
import os
from typing import Callable, List, Optional

import mlx.core as mx
from PIL import Image

from roop.diffusion import hub
from roop.diffusion.registry import (
    ARCH_FLUX_KONTEXT,
    ARCH_SD15,
    ARCH_SDXL,
    ModelSpec,
)

ProgressFn = Optional[Callable[[float, str], None]]

_LOADED_KEY: Optional[str] = None
_LOADED = None


class UnsupportedArchitecture(RuntimeError):
    pass


# diffusers pipeline classes we know how to run, and the arch they map to.
_KNOWN_CLASSES = {
    "StableDiffusionPipeline": ARCH_SD15,
    "StableDiffusionImg2ImgPipeline": ARCH_SD15,
    "StableDiffusionXLPipeline": ARCH_SDXL,
    "StableDiffusionXLImg2ImgPipeline": ARCH_SDXL,
    "FluxPipeline": ARCH_FLUX_KONTEXT,
    "FluxKontextPipeline": ARCH_FLUX_KONTEXT,
}


def detect_arch(root: str, spec: ModelSpec) -> str:
    """Confirm the registry's guess against the repo's own model_index.json.

    Matters for models whose contents we could not inspect ahead of time --
    a private repo can turn out to be something this tab cannot run, and a
    clear error beats a stack trace from a shape mismatch.
    """
    path = os.path.join(root, "model_index.json")
    if not os.path.exists(path):
        return spec.arch
    with open(path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    cls = index.get("_class_name")
    if cls is None:
        return spec.arch
    arch = _KNOWN_CLASSES.get(cls)
    if arch is None:
        raise UnsupportedArchitecture(
            f"{spec.repo} is a {cls}, which this tab does not implement. "
            f"Supported: {', '.join(sorted(_KNOWN_CLASSES))}."
        )
    return arch


def unload() -> None:
    global _LOADED, _LOADED_KEY
    _LOADED = None
    _LOADED_KEY = None
    gc.collect()
    mx.clear_cache()


def load(spec: ModelSpec, progress: ProgressFn = None):
    """Return a ready pipeline for *spec*, fetching and loading if needed."""
    global _LOADED, _LOADED_KEY
    if _LOADED_KEY == spec.key and _LOADED is not None:
        return _LOADED

    unload()
    root = hub.fetch(spec, progress)
    arch = detect_arch(root, spec)

    if arch in (ARCH_SD15, ARCH_SDXL):
        from roop.diffusion.pipeline_sd import SDPipeline
        # Honour the detected arch over the registry's assumption.
        from dataclasses import replace
        pipe = SDPipeline(replace(spec, arch=arch), root, progress)
    elif arch == ARCH_FLUX_KONTEXT:
        from roop.diffusion.pipeline_flux import FluxKontextPipeline
        pipe = FluxKontextPipeline(spec, root, progress)
    else:
        raise UnsupportedArchitecture(f"unknown architecture {arch!r}")

    _LOADED, _LOADED_KEY = pipe, spec.key
    return pipe


def generate(spec: ModelSpec, progress: ProgressFn = None,
             **params) -> List[Image.Image]:
    pipe = load(spec, progress)
    return pipe.generate(progress=progress, **params)


def loaded_key() -> Optional[str]:
    return _LOADED_KEY
