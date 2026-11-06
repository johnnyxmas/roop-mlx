"""Core AI backend: routing, parity vs the ONNX reference, fallbacks, threads.

Model tests skip unless Core AI is usable here (macOS 27+, bridge loads,
coreai-onnx installed) and the model is downloaded.  First run converts each
model (cached in models/**/coreai_cache/), so it is slow once.
"""
import os
import threading

import numpy as np
import pytest

from conftest import model_path, ort_cpu_reference


def _coreai_available():
    from roop.coreai import session
    return session.available()


needs_coreai = pytest.mark.skipif(not _coreai_available(), reason="Core AI not available")


def _smooth_feed(path, seed=7):
    """Image-like input in [-1, 1] (all routed models take that range);
    unit-norm vectors for the swappers' source embedding."""
    import cv2
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    rng = np.random.default_rng(seed)
    feed = {}
    for i in ort.InferenceSession(path, providers=["CPUExecutionProvider"]).get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        if len(shape) == 4:
            _, c, h, w = shape
            img = cv2.resize(rng.random((h // 8 + 1, w // 8 + 1, c)).astype(np.float32), (w, h))
            feed[i.name] = (img.reshape(h, w, c) * 2 - 1).transpose(2, 0, 1)[None].astype(np.float32)
        else:
            v = rng.normal(size=shape).astype(np.float32)
            feed[i.name] = v / np.linalg.norm(v)
    return feed


def _psnr(out, ref):
    d = out.astype(np.float64) - ref.astype(np.float64)
    r = float(ref.max() - ref.min()) or 1.0
    return 10 * np.log10(r * r / max(float((d ** 2).mean()), 1e-30))


@needs_coreai
@pytest.mark.parametrize("name,min_db", [
    # The swappers run reduced precision (as on the CoreML EP, 81 dB);
    # the enhancers run effectively exact fp32.
    ("hyperswap_1a_256.onnx", 70), ("hyperswap_1b_256.onnx", 70), ("hyperswap_1c_256.onnx", 70),
    ("GFPGANv1.4.onnx", 100), ("GPEN-BFR-512.onnx", 100),
    ("GPEN-BFR-1024.onnx", 100), ("GPEN-BFR-2048.onnx", 100),
])
def test_routed_model_runs_on_coreai_with_parity(name, min_db):
    path = model_path(name)
    if not os.path.exists(path):
        pytest.skip(f"model not downloaded: {name}")
    from roop.coreai.session import CoreAISession
    from roop.inference import create_session

    sess = create_session(path)
    assert isinstance(sess, CoreAISession)
    feed = _smooth_feed(path)
    out = sess.run(None, feed)
    ref_first = ort_cpu_reference(path, feed)
    assert out[0].shape == ref_first.shape
    db = _psnr(out[0], ref_first)
    assert db > min_db, f"{name}: {db:.1f} dB vs ORT CPU"


def test_unavailable_falls_back(monkeypatch):
    """No Core AI: the model loads on the backend it used before."""
    path = model_path("GFPGANv1.4.onnx")
    if not os.path.exists(path):
        pytest.skip("model not downloaded")
    from roop.coreai import session
    from roop.inference import create_session
    monkeypatch.setattr(session, "_available", False)
    assert type(create_session(path)).__name__ == "OnnxMLXSession"


@needs_coreai
def test_failed_conversion_falls_back(monkeypatch):
    path = model_path("hyperswap_1a_256.onnx")
    if not os.path.exists(path):
        pytest.skip("model not downloaded")
    from roop.coreai import session
    from roop.inference import create_session
    monkeypatch.setattr(session, "ensure_aimodel", lambda p: None)
    assert type(create_session(path)).__name__ == "InferenceSession"   # CoreML EP


def test_requires_macos_27(monkeypatch):
    from roop.coreai import session
    monkeypatch.setattr(session.platform, "mac_ver", lambda: ("26.4", ("", "", ""), "arm64"))
    monkeypatch.delenv("ROOP_DISABLE_COREAI", raising=False)
    assert session._check_available() is False


def test_env_var_disables(monkeypatch):
    from roop.coreai import session
    monkeypatch.setenv("ROOP_DISABLE_COREAI", "1")
    assert session._check_available() is False


@needs_coreai
def test_concurrent_runs_match_serial():
    """Regression: concurrent runs on one function intermittently corrupted
    output (GPEN-2048: 1 in 40 off by up to 1.82) until runs were serialised
    per session."""
    path = model_path("GPEN-BFR-512.onnx")
    if not os.path.exists(path):
        pytest.skip("model not downloaded")
    from roop.inference import create_session
    sess = create_session(path)
    feed = _smooth_feed(path)
    ref = sess.run(None, feed)[0]
    results = []

    def worker():
        for _ in range(5):
            results.append(sess.run(None, feed)[0])
    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 20
    assert all(np.array_equal(r, ref) for r in results)
