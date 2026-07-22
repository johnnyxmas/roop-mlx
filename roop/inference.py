"""Per-model inference backend selection for Apple Silicon.

Two backends expose the same session API surface (get_inputs /
get_outputs / run):

- onnxruntime with the CoreML execution provider (GPU/ANE via MLProgram)
- OnnxMLXSession, the in-repo ONNX-to-MLX graph interpreter

Benchmarked head-to-head (M-series, 2026-07): CoreML is 2-4x faster for
the detector stack (yoloface, ArcFace, 2d106det) and the large enhancers
(GFPGAN, RestoreFormer++, CodeFormer), and matches the ONNX reference
where the interpreter has measurable numerical drift on the transformer
enhancers.  The interpreter stays faster for GPEN and XSeg, and
inswapper_128 uses the hand-ported native MLX model instead of either.
"""

import os

_COREML_MODELS = {
    "yoloface_8n.onnx",
    "w600k_r50.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "GFPGANv1.4.onnx",
    "restoreformer_plus_plus.onnx",
    "CodeFormerv0.1.onnx",
    "inswapper_128.onnx",
    "hyperswap_1a_256.onnx",
    "hyperswap_1b_256.onnx",
    "hyperswap_1c_256.onnx",
    "reswapper_128.onnx",
    "reswapper_256.onnx",
    # Frame upscaler: SPAN is 2.5x faster on CoreML than the MLX interpreter
    # (2.2 vs 5.4 ms/tile) with 1.6e-5 parity; the static 128x128 tile
    # pinning (_STATIC_DIM_SIZES) keeps it off the E5RT unbounded-dim path.
    # real_esrgan_x2, deoldify and isnet stay on the MLX interpreter — x2
    # partitions poorly on CoreML (5x slower) and the other two fail CoreML
    # compilation outright.  (real_esrgan_x4/lsdir_x4 retired 2026-07:
    # span_kendata_x4 beat both on quality and speed.)
    "span_kendata_x4.onnx",
}

# Models that ran fp16 on the MLX interpreter before moving to CoreML;
# keep that dtype if the CoreML session ever falls back to MLX.
_MLX_FP16_MODELS = {
    "hyperswap_1a_256.onnx",
    "hyperswap_1b_256.onnx",
    "hyperswap_1c_256.onnx",
    "reswapper_128.onnx",
    "reswapper_256.onnx",
}


# Fixed sizes for symbolic input dims, per model.  The tile-based frame
# upscalers always see exactly 128x128 tiles (Frame_Upscale.Run), so their
# spatial dims can be pinned along with the batch dim.
_STATIC_DIM_SIZES = {
    "span_kendata_x4.onnx": {"batch_size": 1, "height": 128, "width": 128},
}


def _static_model_path(model_path: str) -> str:
    """Return a path to *model_path* with its symbolic input dims pinned.

    CoreML's MLProgram runtime rejects unbounded dimensions -- a model with a
    dynamic dim raises "E5RT ... has unbounded dimension" at run time and the
    EP silently partitions it to CPU, forfeiting the ANE/GPU acceleration it
    was routed to CoreML for.  The detector sub-models (w600k_r50, 2d106det,
    genderage) ship a dynamic batch dim yet are only ever called one face at a
    time (see face_detector.py), so pinning N=1 is exact; the frame upscalers
    additionally get their spatial dims pinned via _STATIC_DIM_SIZES.  Symbolic
    dims in the batch position (dim 0) default to 1; a model with an unmapped
    symbolic dim elsewhere is passed through unchanged rather than guessed at.
    The rewritten copy is cached next to the compiled-model cache and
    regenerated if the source changes; already-static models pass straight
    through.
    """
    import onnx

    model = onnx.load(model_path)
    sizes = dict(_STATIC_DIM_SIZES.get(os.path.basename(model_path), {}))
    unmapped = set()
    for inp in model.graph.input:
        for pos, d in enumerate(inp.type.tensor_type.shape.dim):
            if not d.dim_param or d.dim_param in sizes:
                continue
            if pos == 0:
                sizes[d.dim_param] = 1  # batch dim
            else:
                unmapped.add(d.dim_param)
    if unmapped:
        print(
            f"Not pinning {os.path.basename(model_path)} for CoreML: "
            f"non-batch symbolic dims {sorted(unmapped)} have no fixed size."
        )
        return model_path
    if not sizes:
        return model_path

    cache_dir = os.path.join(os.path.dirname(model_path), "coreml_cache")
    fixed = os.path.join(cache_dir, os.path.basename(model_path) + ".static.onnx")
    if os.path.exists(fixed) and os.path.getmtime(fixed) >= os.path.getmtime(model_path):
        return fixed

    from onnxruntime.tools.onnx_model_utils import (
        fix_output_shapes,
        make_dim_param_fixed,
    )

    for param, value in sizes.items():
        make_dim_param_fixed(model.graph, param, value)
    fix_output_shapes(model)
    os.makedirs(cache_dir, exist_ok=True)
    onnx.save(model, fixed)
    return fixed


def _create_coreml_session(model_path: str):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    # Cache compiled MLPrograms next to the model so only the first launch
    # pays CoreML compilation (up to ~20s for CodeFormer).
    cache_dir = os.path.join(os.path.dirname(model_path), "coreml_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return ort.InferenceSession(
        _static_model_path(model_path),
        providers=[
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": cache_dir,
            }),
            "CPUExecutionProvider",
        ],
    )


def prewarm_sessions_async():
    """Warm the CoreML compile caches for the always-used models.

    Runs in a daemon thread at app startup.  Sessions are created and
    discarded; the win is CoreML's on-disk compiled-model cache (keyed by
    model path), so the first real swap doesn't stall on compilation.
    """
    import threading

    from roop.utilities import resolve_relative_path

    def _warm():
        for rel in (
            "../models/yoloface_8n.onnx",
            "../models/buffalo_l/w600k_r50.onnx",
            "../models/buffalo_l/2d106det.onnx",
            "../models/inswapper_128.onnx",
        ):
            try:
                path = resolve_relative_path(rel)
                if os.path.exists(path):
                    _create_coreml_session(path)
            except Exception:
                return

    threading.Thread(target=_warm, daemon=True, name="coreml-prewarm").start()


def try_coreml_session(model_path: str):
    """Return a CoreML EP session, or None if unavailable."""
    try:
        return _create_coreml_session(model_path)
    except Exception as e:
        print(f"CoreML session failed for {os.path.basename(model_path)} ({e})")
        return None


def create_session(model_path: str):
    """Return the fastest verified inference session for *model_path*."""
    name = os.path.basename(model_path)
    if name in _COREML_MODELS:
        sess = try_coreml_session(model_path)
        if sess is not None:
            return sess
        print(f"Falling back to MLX interpreter for {name}")
    from roop.mlx_runtime.onnx_loader import OnnxMLXSession
    if name in _MLX_FP16_MODELS:
        import mlx.core as mx
        return OnnxMLXSession(model_path, dtype=mx.float16)
    return OnnxMLXSession(model_path)
