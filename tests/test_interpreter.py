"""OnnxMLXSession interpreter parity for the models that run on it in
production (GPEN, XSeg, DeOldify, ISNet, Real-ESRGAN x2) plus regression
coverage for the opset-10 Resize bug that silently disabled upsampling."""

import gc

import numpy as np
import pytest

from conftest import make_feed, model_path, ort_cpu_reference, requires_model


def _interpreter_output(path, feed, outs=None):
    import mlx.core as mx
    from roop.mlx_runtime.onnx_loader import OnnxMLXSession

    sess = OnnxMLXSession(path)
    out_names = outs or [sess.get_outputs()[0].name]
    out = sess.run(out_names, feed)[0]
    del sess
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return out


@pytest.mark.parametrize("parts,outs,tol_max", [
    (("GPEN-BFR-512.onnx",), ["output"], 0.05),
    (("xseg.onnx",), None, 0.05),
    (("Frame", "isnet-general-use.onnx"), None, 0.01),
])
def test_interpreter_parity(parts, outs, tol_max):
    import os
    path = model_path(*parts)
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {parts}")
    feed = make_feed(path)
    ref = ort_cpu_reference(path, feed, outs)
    out = _interpreter_output(path, feed, outs)
    d = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    assert d.max() < tol_max, f"{parts[-1]} drifted: max {d.max()}"


@requires_model("Frame", "real_esrgan_x2.onnx")
def test_opset10_resize_upscales():
    """Regression: opset-10 Resize ([X, scales]) was silently treated as
    identity, so Real-ESRGAN produced a HALF-size output instead of 2x."""
    path = model_path("Frame", "real_esrgan_x2.onnx")
    rng = np.random.default_rng(7)
    feed = {"input": rng.random((1, 3, 128, 128)).astype(np.float32)}
    ref = ort_cpu_reference(path, feed)
    out = _interpreter_output(path, feed)
    assert out.shape == (1, 3, 256, 256), f"wrong output shape {out.shape}"
    d = np.abs(out - ref)
    assert d.max() < 0.02, f"esrgan_x2 drifted: max {d.max()}"
