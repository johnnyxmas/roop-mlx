"""Registry of the diffusion models exposed by the Stable Diffusion tab.

Nothing here downloads anything.  A model is a pure description -- HF repo,
architecture family, which weight files are needed and which UI controls are
meaningful for it -- until the user actually presses Generate, at which point
roop.diffusion.hub fetches it.

The `params` / `overrides` fields drive the tab: the UI builds one superset of
controls and shows/retunes the subset each model declares, so adding a model
here is enough to give it a correct control panel.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# -- parameter keys the UI knows how to render ------------------------------
# Keep these in sync with ui/tabs/diffusion_tab.py::_CONTROLS.
P_PROMPT = "prompt"
P_NEGATIVE = "negative_prompt"
P_STEPS = "steps"
P_GUIDANCE = "guidance"
P_TRUE_CFG = "true_cfg"
P_SEED = "seed"
P_WIDTH = "width"
P_HEIGHT = "height"
P_STRENGTH = "strength"
P_SAMPLER = "sampler"
P_CLIP_SKIP = "clip_skip"
P_BATCH = "batch"

# Architecture families -- each maps to one pipeline in roop.diffusion.pipeline.
ARCH_SD15 = "sd15"
ARCH_SDXL = "sdxl"
ARCH_FLUX_KONTEXT = "flux_kontext"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    arch: str
    #: Human note shown under the dropdown.
    blurb: str
    #: Approximate on-disk size once fetched, for the pre-download prompt.
    download_gb: float
    #: Resolution the model was trained at; the UI clamps width/height to a
    #: sensible band around it.
    native_size: int
    #: Which UI controls apply.
    params: List[str]
    #: Per-control overrides: {param_key: {"value": .., "minimum": .., ...}}
    overrides: Dict[str, dict] = field(default_factory=dict)
    #: True when the HF repo needs an accepted licence / auth token.
    gated: bool = False
    #: Prefer these variant-suffixed weight files when present (halves the
    #: download for repos that publish an fp16 mirror).
    variant: str = ""
    #: True when the model edits an input image rather than only generating.
    needs_image: bool = False


_SD_SAMPLERS = ["Euler", "Euler Ancestral", "DDIM"]

MODELS: List[ModelSpec] = [
    ModelSpec(
        key="realistic_vision",
        label="RealisticVision 5.1 (SD 1.5)",
        repo="SG161222/Realistic_Vision_V5.1_noVAE",
        arch=ARCH_SD15,
        blurb="Photoreal SD 1.5 finetune. Fast, 512px native. Good for "
              "photographic img2img at low strength.",
        download_gb=4.3,
        native_size=512,
        params=[P_PROMPT, P_NEGATIVE, P_STEPS, P_GUIDANCE, P_SEED, P_WIDTH,
                P_HEIGHT, P_STRENGTH, P_SAMPLER, P_CLIP_SKIP, P_BATCH],
        overrides={
            P_GUIDANCE: {"value": 7.0},
            P_STEPS: {"value": 25},
            P_CLIP_SKIP: {"value": 2},
            P_NEGATIVE: {
                "value": "(deformed iris, deformed pupils, semi-realistic, cgi, "
                         "3d, render, sketch, cartoon, drawing, anime), text, "
                         "cropped, out of frame, worst quality, low quality, "
                         "jpeg artifacts, ugly, duplicate, mutilated, extra "
                         "fingers, mutated hands, poorly drawn hands, poorly "
                         "drawn face, mutation, deformed, blurry"
            },
        },
    ),
    ModelSpec(
        key="dreamshaper",
        label="DreamShaper 8 (SD 1.5)",
        repo="Lykon/dreamshaper-8",
        arch=ARCH_SD15,
        blurb="Versatile SD 1.5 finetune -- handles photo, illustration and "
              "concept-art styles from the same prompt vocabulary.",
        download_gb=2.2,
        native_size=512,
        variant="fp16",
        params=[P_PROMPT, P_NEGATIVE, P_STEPS, P_GUIDANCE, P_SEED, P_WIDTH,
                P_HEIGHT, P_STRENGTH, P_SAMPLER, P_CLIP_SKIP, P_BATCH],
        overrides={
            P_GUIDANCE: {"value": 7.5},
            P_STEPS: {"value": 25},
            P_NEGATIVE: {"value": "worst quality, low quality, blurry, "
                                  "watermark, text, deformed"},
        },
    ),
    ModelSpec(
        key="sdxl",
        label="SDXL 1.0 base",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        arch=ARCH_SDXL,
        blurb="1024px base model with dual text encoders. Slower per step "
              "than SD 1.5 but much stronger composition and text adherence.",
        download_gb=6.9,
        native_size=1024,
        variant="fp16",
        params=[P_PROMPT, P_NEGATIVE, P_STEPS, P_GUIDANCE, P_SEED, P_WIDTH,
                P_HEIGHT, P_STRENGTH, P_SAMPLER, P_BATCH],
        overrides={
            P_GUIDANCE: {"value": 6.0},
            P_STEPS: {"value": 30},
            P_WIDTH: {"value": 1024, "minimum": 512, "maximum": 1536},
            P_HEIGHT: {"value": 1024, "minimum": 512, "maximum": 1536},
        },
    ),
    ModelSpec(
        key="flux_kontext",
        label="FLUX.1 Kontext [dev] (instruction edit)",
        repo="black-forest-labs/FLUX.1-Kontext-dev",
        arch=ARCH_FLUX_KONTEXT,
        blurb="Instruction-driven editing: describe the change ('make the "
              "jacket red'), not the whole scene. Needs an input image. "
              "Gated repo -- accept the licence on HuggingFace and set "
              "HF_TOKEN first. Large download.",
        download_gb=32.0,
        native_size=1024,
        gated=True,
        needs_image=True,
        params=[P_PROMPT, P_STEPS, P_GUIDANCE, P_SEED, P_WIDTH, P_HEIGHT,
                P_BATCH],
        overrides={
            # Kontext is a guidance-distilled flow model: `guidance` is an
            # embedded conditioning scalar, not classifier-free guidance, so
            # the usable band is far narrower than SD's.
            P_GUIDANCE: {"value": 2.5, "minimum": 1.0, "maximum": 10.0,
                         "step": 0.1,
                         "label": "Guidance (embedded, not CFG)"},
            P_STEPS: {"value": 28, "maximum": 50},
            P_WIDTH: {"value": 1024, "minimum": 512, "maximum": 1536},
            P_HEIGHT: {"value": 1024, "minimum": 512, "maximum": 1536},
        },
    ),
]

# Deliberately absent: Pruna's P-Image-Edit (`prunaai/p-image-edit`).  That is
# a Replicate endpoint, not a HuggingFace repo -- the base model is served
# only from Pruna's own infrastructure and has no published weights.  The
# LoRAs they do publish on HuggingFace are not safetensors either: the files
# carry a `REPLORA1` magic and an encrypted body, so they only decrypt inside
# Replicate.  Nothing about it can run on-device; FLUX.1 Kontext above covers
# instruction editing instead.

BY_KEY: Dict[str, ModelSpec] = {m.key: m for m in MODELS}
BY_LABEL: Dict[str, ModelSpec] = {m.label: m for m in MODELS}

LABELS: List[str] = [m.label for m in MODELS]


def get(label_or_key: str) -> ModelSpec:
    if label_or_key in BY_LABEL:
        return BY_LABEL[label_or_key]
    return BY_KEY[label_or_key]


def samplers_for(spec: ModelSpec) -> List[str]:
    """Sampler choices valid for *spec* (flow models have a single scheduler)."""
    if spec.arch == ARCH_FLUX_KONTEXT:
        return ["Flow Match Euler"]
    return list(_SD_SAMPLERS)
