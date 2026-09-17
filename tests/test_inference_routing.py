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
    (("GFPGANv1.4.onnx",),               "OnnxMLXSession"),
    (("alphaface_256.onnx",),            "InferenceSession"),
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
    """Tile-based upscalers must pin batch=1 AND spatial dims to the tile
    size Frame_Upscale feeds this model — pinning spatial dims to 1 (the
    batch default) would break tile inference at run time."""
    import os

    import numpy as np
    import onnx

    path = model_path("Frame", fname)
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {fname}")
    from roop.inference import _static_model_path
    from roop.processors.Frame_Upscale import _MODEL_CONFIGS

    (tile,) = {
        cfg[2] for cfg in _MODEL_CONFIGS.values() if cfg[0] == fname
    }
    static = _static_model_path(path)
    assert static != path
    model = onnx.load(static, load_external_data=False)
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, tile, tile], dims

    # and the static graph must actually accept a tile of that size
    import onnxruntime as ort
    sess = ort.InferenceSession(static, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.zeros((1, 3, tile, tile), np.float32)})[0]
    assert out.shape == (1, 3, tile * 4, tile * 4)


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
        # CoreML EP with MLX-interpreter fallback (native model retired)
        assert _backend_name(p._session) in ("InferenceSession", "OnnxMLXSession")
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


def test_prune_removes_unreachable_cache_entries(tmp_path):
    """Launch/shutdown sweep: legacy layout and orphans go, live entries stay."""
    from roop.inference import prune_compile_caches

    models = tmp_path / "models"
    cache = models / "coreml_cache"
    models.mkdir()
    (models / "live.onnx").write_bytes(b"x" * 10)

    # Live entries: compiled dir, marker and dim-pinned rewrite for a model
    # that is still on disk.
    (cache / "live.onnx").mkdir(parents=True)
    (cache / "live.onnx" / "model.mil").write_bytes(b"y" * 100)
    (cache / "live.onnx.nocache").write_text("1.28.0 27.0")
    (cache / "live.onnx.static.1.onnx").write_bytes(b"z" * 50)

    # Garbage: pre-2026-09 hash dir, a nested cache root, and entries whose
    # model was retired.
    (cache / "1234567890123456789").mkdir()
    (cache / "1234567890123456789" / "model.mil").write_bytes(b"w" * 1000)
    (cache / "coreml_cache" / "987654321").mkdir(parents=True)
    (cache / "retired.onnx").mkdir()
    (cache / "retired.onnx.nocache").write_text("1.28.0 27.0")
    (cache / "retired.onnx.static.1.onnx").write_bytes(b"q" * 30)

    freed = prune_compile_caches(str(models))

    assert freed > 0
    survivors = sorted(p.name for p in cache.iterdir())
    assert survivors == ["live.onnx", "live.onnx.nocache", "live.onnx.static.1.onnx"]
    assert (cache / "live.onnx" / "model.mil").exists()


def test_prune_is_a_noop_on_a_clean_tree(tmp_path):
    from roop.inference import prune_compile_caches

    models = tmp_path / "models"
    (models / "coreml_cache" / "live.onnx").mkdir(parents=True)
    (models / "live.onnx").write_bytes(b"x")

    assert prune_compile_caches(str(models)) == 0
    assert (models / "coreml_cache" / "live.onnx").exists()
