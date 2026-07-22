"""Swap model numerical parity against the ONNX reference (ORT CPU EP).

Tolerances are anchored to measured values from the 2026-07 backend
benchmarks, with headroom.  A failure here means a backend regressed
numerically, not just stylistically.
"""

import numpy as np
import pytest

from conftest import make_feed, model_path, ort_cpu_reference, requires_model


@requires_model("inswapper_128.onnx")
def test_inswapper_coreml_bit_parity():
    """CoreML EP matched the reference bit-for-bit when adopted."""
    path = model_path("inswapper_128.onnx")
    feed = make_feed(path)
    ref = ort_cpu_reference(path, feed, ["output"])

    from roop.inference import try_coreml_session

    sess = try_coreml_session(path)
    if sess is None:
        pytest.skip("CoreML unavailable")
    out = sess.run(["output"], feed)[0]
    d = np.abs(out - ref)
    assert d.max() < 1e-3, f"inswapper CoreML drifted: max {d.max()}"


@pytest.mark.parametrize("fname,tol_max", [
    ("hyperswap_1a_256.onnx", 0.03),
    ("reswapper_128.onnx", 0.03),
])
def test_other_swappers_coreml_parity(fname, tol_max):
    import os
    path = model_path(fname)
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {fname}")
    feed = make_feed(path)
    ref = ort_cpu_reference(path, feed)

    from roop.inference import try_coreml_session

    sess = try_coreml_session(path)
    if sess is None:
        pytest.skip("CoreML unavailable")
    outn = [sess.get_outputs()[0].name]
    out = sess.run(outn, feed)[0]
    d = np.abs(out.astype(np.float32) - ref)
    assert d.max() < tol_max, f"{fname} CoreML drifted: max {d.max()}"
