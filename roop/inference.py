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


def _create_coreml_session(model_path: str):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    # Cache compiled MLPrograms next to the model so only the first launch
    # pays CoreML compilation (up to ~20s for CodeFormer).
    cache_dir = os.path.join(os.path.dirname(model_path), "coreml_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return ort.InferenceSession(
        model_path,
        providers=[
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": cache_dir,
            }),
            "CPUExecutionProvider",
        ],
    )


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
