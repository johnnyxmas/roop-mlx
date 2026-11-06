"""OnnxMLXSession interpreter parity for the models that run on it in
production (GPEN, XSeg, DeOldify, ISNet, Real-ESRGAN x2) plus regression
coverage for the opset-10 Resize bug that silently disabled upsampling."""

import gc
import os
import tempfile

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


@pytest.mark.parametrize("mode", ["constant", "reflect", "edge"])
def test_pad_modes_match_onnx(mode):
    """Regression: 'edge' Pad fell through to the reflect implementation.

    Asymmetric pads on both spatial axes, like DeOldify's ReplicationPad.
    Built in memory, so this runs without any downloaded model.
    """
    import onnx
    from onnx import TensorProto, helper

    pads = np.array([0, 0, 2, 1, 0, 0, 1, 3], dtype=np.int64)
    graph = helper.make_graph(
        [helper.make_node("Pad", ["x", "pads"], ["y"], mode=mode)],
        "pad",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 5, 6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [helper.make_tensor("pads", TensorProto.INT64, [8], pads)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pad.onnx")
        onnx.save(model, path)
        x = np.random.default_rng(0).normal(size=(1, 2, 5, 6)).astype(np.float32)
        ref = ort_cpu_reference(path, {"x": x})
        out = _interpreter_output(path, {"x": x})
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("name", ["deoldify_artistic.onnx", "deoldify_stable.onnx"])
def test_deoldify_parity(name):
    """DeOldify at the scale Frame_Colorizer really feeds it (raw 0-255 RGB,
    256x256; the generic random feed would pin its dynamic H/W to 1).

    With 'edge' padding mis-run as reflect, this feed drifted 13.8 (artistic)
    and 12.3 (stable) on the 0-255 output — a visible colour shift.  Fixed,
    both measure 0.0125.
    """
    import os
    path = model_path("Frame", name)
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {name}")
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (256, 256)).astype(np.float32)
    # Smooth it so the model sees image-like structure, not white noise.
    import cv2
    gray = cv2.GaussianBlur(gray, (0, 0), 3)
    feed = {"input": np.repeat(gray[None, None], 3, axis=1)}
    ref = ort_cpu_reference(path, feed)
    out = _interpreter_output(path, feed)
    d = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    assert d.max() < 0.5, f"{name} drifted: max {d.max()} (0-255 scale)"


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


@requires_model("GFPGANv1.4.onnx")
def test_gfpgan_interpreter_parity():
    """GFPGAN was routed off CoreML 2026-09-16 for a 28% speedup (222 vs 309 ms).

    Its interpreter drift is real but small, and larger than GPEN's: worst
    single pixel 0.056 of the [-1,1] output range on a white-noise feed
    (0.063 on face-like input), mean 1.5e-3.  In 8-bit terms that is ~7/255
    at one pixel and 0.2/255 on average -- 58 dB PSNR, better than the fp16
    inference this repo already ships for the 256px swappers.  Asserting on
    PSNR rather than a loosened max keeps the guard meaningful: the drift
    that forced CodeFormer (1.37) and RestoreFormer++ (3.58) onto CoreML
    would fail this by a mile.

    2026-11-05: nearly all of that drift was MLX's winograd conv running
    fp32 at TF32 precision on M5.  With it disabled (onnx_loader.py sets
    MLX_CONV_WINOGRAD=0) this feed measures 122 dB, max 8e-5, so the bounds
    below are tight enough to catch winograd being switched back on.
    """
    path = model_path("GFPGANv1.4.onnx")
    feed = make_feed(path)
    ref = ort_cpu_reference(path, feed).astype(np.float64)
    out = _interpreter_output(path, feed).astype(np.float64)

    to8 = lambda a: np.clip((a + 1) * 127.5, 0, 255)  # noqa: E731
    psnr = 10 * np.log10(255 ** 2 / ((to8(ref) - to8(out)) ** 2).mean())

    assert psnr > 100, f"GFPGAN interpreter parity regressed: {psnr:.1f} dB"
    assert np.abs(ref - out).max() < 1e-3
