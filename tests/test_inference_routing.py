"""Backend routing: each model must load on its benchmarked-fastest backend."""

import pytest

from conftest import model_path, requires_model


def _backend_name(session) -> str:
    return type(session).__name__


@pytest.mark.parametrize("parts,expected", [
    (("yoloface_8n.onnx",),              "InferenceSession"),
    (("buffalo_l", "w600k_r50.onnx"),    "InferenceSession"),
    (("buffalo_l", "2d106det.onnx"),     "InferenceSession"),
    (("buffalo_l", "genderage.onnx"),    "InferenceSession"),
    (("Frame", "span_kendata_x4.onnx"),  "InferenceSession"),
    (("Frame", "real_esrgan_x2.onnx"),   "OnnxMLXSession"),
    (("GPEN-BFR-512.onnx",),             "OnnxMLXSession"),
    (("xseg.onnx",),                     "OnnxMLXSession"),
])
def test_create_session_backend(parts, expected):
    path = model_path(*parts)
    import os
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {parts}")
    from roop.inference import create_session

    sess = create_session(path)
    assert _backend_name(sess) == expected


@pytest.mark.parametrize("parts", [
    ("buffalo_l", "w600k_r50.onnx"),
    ("buffalo_l", "2d106det.onnx"),
    ("buffalo_l", "genderage.onnx"),
])
def test_static_model_has_no_dynamic_dims(parts):
    """CoreML's MLProgram runtime rejects unbounded dims (E5RT).  The detector
    sub-models ship a dynamic batch dim, so the CoreML path must rewrite them to
    a fully static graph before handing them to the EP."""
    import os

    import onnx

    path = model_path(*parts)
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {parts}")
    from roop.inference import _static_model_path

    static = _static_model_path(path)
    assert static != path, "model with a dynamic dim should be rewritten"

    model = onnx.load(static, load_external_data=False)
    dynamic = [
        (inp.name, d.dim_param)
        for inp in model.graph.input
        for d in inp.type.tensor_type.shape.dim
        if d.dim_param
    ]
    assert dynamic == [], f"static copy still has symbolic dims: {dynamic}"
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1


@pytest.mark.parametrize("fname", [
    "span_kendata_x4.onnx",
])
def test_static_upscaler_pins_tile_size(fname):
    """Tile-based upscalers must pin batch=1 AND spatial dims to the 128px
    tile Frame_Upscale always feeds — pinning spatial dims to 1 (the batch
    default) would break tile inference at run time."""
    import os

    import numpy as np
    import onnx

    path = model_path("Frame", fname)
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {fname}")
    from roop.inference import _static_model_path

    static = _static_model_path(path)
    assert static != path
    model = onnx.load(static, load_external_data=False)
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, 128, 128], dims

    # and the static graph must actually accept a 128px tile
    import onnxruntime as ort
    sess = ort.InferenceSession(static, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.zeros((1, 3, 128, 128), np.float32)})[0]
    assert out.shape == (1, 3, 512, 512)


@requires_model("yoloface_8n.onnx")
def test_static_model_path_passthrough_for_static_model():
    """Already-static models must not be needlessly copied."""
    path = model_path("yoloface_8n.onnx")
    from roop.inference import _static_model_path

    assert _static_model_path(path) == path


@requires_model("inswapper_128.onnx")
def test_inswapper_processor_uses_coreml():
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace

    p = FaceSwapInsightFace()
    p.Initialize({"modelname": "inswapper_128.onnx", "devicename": "mps", "device_id": 0})
    try:
        assert _backend_name(p._session) in ("InferenceSession", "InSwapperNative")
        if _backend_name(p._session) == "InferenceSession":
            # CoreML path cannot batch (graph declares batch=1)
            assert p.supports_batch is False
        else:
            # native fallback batches pixel-boost slices
            assert p.supports_batch is True
    finally:
        p.Release()


@requires_model("hyperswap_1b_256.onnx")
def test_hyperswap_processor_session():
    from roop.processors.FaceSwapFaceFusion import FaceSwapFaceFusion

    p = FaceSwapFaceFusion()
    p.Initialize({"modelname": "HyperSwap 1B"})
    try:
        assert _backend_name(p._session) in ("InferenceSession", "OnnxMLXSession")
        assert p.swap_type == "hyperswap"
    finally:
        p.Release()
