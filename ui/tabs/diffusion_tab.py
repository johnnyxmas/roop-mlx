"""Stable Diffusion tab -- prompt-driven generation and image editing.

The control panel is generated from roop.diffusion.registry: each model
declares which parameters it understands and how they should be tuned, and
switching models reconfigures the panel rather than showing a lowest-common
-denominator form.  Nothing is downloaded until Generate is pressed.
"""

import os
import time
from typing import List

import gradio as gr

import roop.globals
from roop.diffusion import pipeline, registry
from roop.diffusion.hub import ModelAccessError, is_cached
from roop.diffusion.registry import (
    P_BATCH,
    P_CLIP_SKIP,
    P_GUIDANCE,
    P_HEIGHT,
    P_NEGATIVE,
    P_PROMPT,
    P_SAMPLER,
    P_SEED,
    P_STEPS,
    P_STRENGTH,
    P_WIDTH,
    ModelSpec,
)

# Order the handlers pass controls in; must match _controls() and the
# component list wired to the model dropdown.
_CONTROL_ORDER = [P_PROMPT, P_NEGATIVE, P_STEPS, P_GUIDANCE, P_SAMPLER,
                  P_WIDTH, P_HEIGHT, P_STRENGTH, P_CLIP_SKIP, P_SEED, P_BATCH]


def _describe(spec: ModelSpec) -> str:
    if is_cached(spec):
        state = "✅ **Downloaded** — ready to run."
    else:
        state = (f"⬇️ **Not downloaded yet** (~{spec.download_gb:.0f} GB). "
                 "The first Generate will fetch it.")
        if spec.gated:
            state += " Gated repo: accept the licence and set `HF_TOKEN` first."
    return f"{spec.blurb}\n\n{state}\n\n`{spec.repo}`"


def _settings_for(spec: ModelSpec, key: str, **defaults) -> dict:
    """Constructor kwargs for one control under *spec*.

    The tab's baseline values live here rather than being hardcoded twice,
    so the panel as first rendered matches what a switch back to the same
    model would produce.
    """
    kwargs = dict(defaults)
    kwargs["visible"] = key in spec.params
    kwargs.update(spec.overrides.get(key, {}))
    return kwargs


def _update_for(spec: ModelSpec, key: str, **extra):
    """gr.update for one control, applying the model's overrides."""
    return gr.update(**_settings_for(spec, key, **extra))


def _bounds(spec: ModelSpec, key: str) -> tuple:
    o = spec.overrides.get(key, {})
    return o.get("minimum", 256), o.get("maximum", 1536)


def fit_to_model(spec: ModelSpec, width: int, height: int,
                 step: int = 64) -> tuple:
    """Output size matching an upload's aspect, at the model's native scale.

    Two things go wrong with a fixed default. Kontext and SDXL generate a
    fresh canvas at whatever size is asked for, so a portrait photo into a
    square default comes back resampled and soft. And simply copying the
    upload's pixel dimensions is just as bad the other way -- a 4000x3000
    photo would push SD 1.5 far off the resolution it was trained at.

    So: keep the aspect ratio, scale to the model's native pixel *area*, then
    clamp and snap to the slider's step.
    """
    if not width or not height:
        return None, None
    lo, hi = _bounds(spec, P_WIDTH)
    scale = (spec.native_size ** 2 / float(width * height)) ** 0.5
    out = []
    for value in (width * scale, height * scale):
        snapped = int(round(value / step)) * step
        out.append(max(lo, min(hi, snapped)))
    return out[0], out[1]


def on_image_uploaded(image, label: str):
    """Match the output size to the uploaded image when a model uses it."""
    spec = registry.get(label)
    if image is None or P_WIDTH not in spec.params:
        return gr.update(), gr.update()
    size = getattr(image, "size", None)
    if not size:
        return gr.update(), gr.update()
    w, h = fit_to_model(spec, *size)
    if w is None:
        return gr.update(), gr.update()
    return gr.update(value=w), gr.update(value=h)


def diffusion_tab():
    first = registry.MODELS[0]

    with gr.Tab("🎨 Stable Diffusion"):
        with gr.Row():
            with gr.Column(scale=5):
                model_dd = gr.Dropdown(
                    choices=registry.LABELS,
                    value=first.label,
                    label="Model",
                    info="Weights download on first use, not at startup.",
                )
                model_info = gr.Markdown(_describe(first))

                prompt = gr.Textbox(
                    **_settings_for(first, P_PROMPT, label="Prompt", lines=3,
                                    placeholder="What you want to see…"))
                negative = gr.Textbox(
                    **_settings_for(first, P_NEGATIVE,
                                    label="Negative prompt", lines=2,
                                    placeholder="What to avoid…"))

                with gr.Row():
                    steps = gr.Slider(
                        **_settings_for(first, P_STEPS, minimum=1, maximum=100,
                                        value=25, step=1, label="Steps"))
                    guidance = gr.Slider(
                        **_settings_for(first, P_GUIDANCE, minimum=1.0,
                                        maximum=20.0, value=7.0, step=0.1,
                                        label="Guidance (CFG)"))
                with gr.Row():
                    samplers = registry.samplers_for(first)
                    sampler = gr.Dropdown(choices=samplers, value=samplers[0],
                                          label="Sampler",
                                          visible=len(samplers) > 1)
                    clip_skip = gr.Slider(
                        **_settings_for(first, P_CLIP_SKIP, minimum=1,
                                        maximum=4, value=1, step=1,
                                        label="CLIP skip"))
                with gr.Row():
                    width = gr.Slider(
                        **_settings_for(first, P_WIDTH, minimum=256,
                                        maximum=1536, value=512, step=64,
                                        label="Width"))
                    height = gr.Slider(
                        **_settings_for(first, P_HEIGHT, minimum=256,
                                        maximum=1536, value=512, step=64,
                                        label="Height"))
                strength = gr.Slider(
                    **_settings_for(
                        first, P_STRENGTH, minimum=0.05, maximum=1.0,
                        value=0.65, step=0.05,
                        label="Denoise strength (image-to-image)",
                        info="How far from the input image to travel. Only "
                             "used when an input image is supplied."))
                with gr.Row():
                    seed = gr.Number(
                        **_settings_for(first, P_SEED, value=-1, precision=0,
                                        label="Seed", info="-1 for random"))
                    batch = gr.Slider(
                        **_settings_for(first, P_BATCH, minimum=1, maximum=4,
                                        value=1, step=1, label="Images"))

                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    unload_btn = gr.Button("Unload model from memory")

            with gr.Column(scale=5):
                input_image = gr.Image(
                    label="Input image (optional for SD, required for edit "
                          "models)",
                    type="pil", height=320,
                )
                gr.Markdown(
                    "<span>Output width/height follow the uploaded image's "
                    "aspect ratio at the model's native scale — override them "
                    "below if you want a different framing.</span>"
                )
                output_gallery = gr.Gallery(label="Output", height=460,
                                            columns=2, object_fit="contain")
                status = gr.Markdown("")

        controls = [prompt, negative, steps, guidance, sampler, width, height,
                    strength, clip_skip, seed, batch]

        model_dd.change(
            fn=on_model_changed, inputs=[model_dd, input_image],
            outputs=[model_info] + controls,
        )
        # Editing models render a fresh canvas at whatever size is requested,
        # so a portrait photo left at a square default comes back resampled.
        input_image.change(
            fn=on_image_uploaded, inputs=[input_image, model_dd],
            outputs=[width, height],
        )

        # -- generation ---------------------------------------------------
        generate_btn.click(
            fn=on_generate,
            inputs=[model_dd, input_image] + controls,
            outputs=[output_gallery, status, model_info],
        )
        unload_btn.click(fn=on_unload, outputs=[status])


def on_model_changed(label: str, image=None):
    """Reconfigure every control for the newly selected model.

    When an image is already loaded, the size controls follow it rather than
    snapping back to the new model's fixed default.
    """
    spec = registry.get(label)
    samplers = registry.samplers_for(spec)

    size = getattr(image, "size", None) if image is not None else None
    fitted = fit_to_model(spec, *size) if size else (None, None)

    # Built by hand rather than via _update_for: that applies the model's
    # overrides last, so an override's fixed value would win over the size
    # we just fitted to the image.
    width_kwargs = _settings_for(spec, P_WIDTH)
    height_kwargs = _settings_for(spec, P_HEIGHT)
    if fitted[0] is not None and P_WIDTH in spec.params:
        width_kwargs["value"], height_kwargs["value"] = fitted

    return (
        _describe(spec),
        _update_for(spec, P_PROMPT),
        _update_for(spec, P_NEGATIVE),
        _update_for(spec, P_STEPS),
        _update_for(spec, P_GUIDANCE, label="Guidance (CFG)"),
        gr.update(visible=len(samplers) > 1, choices=samplers,
                  value=samplers[0]),
        gr.update(**width_kwargs),
        gr.update(**height_kwargs),
        _update_for(spec, P_STRENGTH),
        _update_for(spec, P_CLIP_SKIP),
        _update_for(spec, P_SEED),
        _update_for(spec, P_BATCH),
    )


def on_unload():
    pipeline.unload()
    return "Model unloaded; memory released."


def on_generate(label, image, prompt_text, negative_text, steps, guidance,
                sampler, width, height, strength, clip_skip, seed, batch,
                progress=gr.Progress()):
    spec = registry.get(label)

    if not (prompt_text or "").strip():
        gr.Warning("Enter a prompt first.")
        return [], "", _describe(spec)
    if spec.needs_image and image is None:
        gr.Warning(f"{spec.label} edits an existing image — add an input "
                   "image.")
        return [], "", _describe(spec)

    params = dict(
        prompt=prompt_text,
        negative_prompt=negative_text or "",
        steps=int(steps), guidance=float(guidance), seed=int(seed),
        width=int(width), height=int(height), strength=float(strength),
        sampler=sampler, clip_skip=int(clip_skip), batch=int(batch),
        image=image,
    )
    # Neutralise anything this model does not expose, so a slider left over
    # from the previously selected model cannot leak into the call.
    for key, neutral in ((P_NEGATIVE, ""), (P_STRENGTH, 1.0),
                         (P_CLIP_SKIP, 1)):
        if key not in spec.params:
            params[key] = neutral

    def report(frac: float, msg: str):
        progress(min(max(frac, 0.0), 1.0), desc=msg)

    started = time.time()
    try:
        images = pipeline.generate(spec, progress=report, **params)
    except ModelAccessError as exc:
        gr.Warning(str(exc))
        return [], f"⚠️ {exc}", _describe(spec)
    except Exception as exc:                      # noqa: BLE001 - surface it
        gr.Warning(f"Generation failed: {exc}")
        return [], f"⚠️ **{type(exc).__name__}:** {exc}", _describe(spec)

    saved = _save_outputs(images, spec)
    elapsed = time.time() - started
    note = (f"Generated {len(images)} image(s) in {elapsed:.1f}s with "
            f"**{spec.label}**. Saved to `{roop.globals.output_path}`.")
    return saved, note, _describe(spec)


def _save_outputs(images, spec: ModelSpec) -> List[str]:
    out_dir = roop.globals.output_path or os.path.abspath("output")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(out_dir, f"{spec.key}_{stamp}_{i}.png")
        img.save(path)
        paths.append(path)
    return paths
