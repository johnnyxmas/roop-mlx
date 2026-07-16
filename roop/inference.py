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
    "GFPGANv1.4.onnx",
    "restoreformer_plus_plus.onnx",
    "CodeFormerv0.1.onnx",
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


def create_session(model_path: str):
    """Return the fastest verified inference session for *model_path*."""
    name = os.path.basename(model_path)
    if name in _COREML_MODELS:
        try:
            return _create_coreml_session(model_path)
        except Exception as e:
            print(f"CoreML session failed for {name} ({e}); using MLX interpreter")
    from roop.mlx_runtime.onnx_loader import OnnxMLXSession
    return OnnxMLXSession(model_path)
