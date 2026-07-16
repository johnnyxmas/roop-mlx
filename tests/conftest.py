"""Shared fixtures for the roop-mlx test suite.

Most tests exercise real models from models/ (downloaded on first app
launch) and skip cleanly when a model file is absent.  Parity tolerances
are anchored to onnxruntime's CPU execution provider, which serves as the
ONNX reference implementation.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO_ROOT, "models")

sys.path.insert(0, REPO_ROOT)


def model_path(*parts) -> str:
    return os.path.join(MODELS, *parts)


def requires_model(*parts):
    """Skip marker for tests that need a downloaded model file."""
    path = model_path(*parts)
    return pytest.mark.skipif(
        not os.path.exists(path), reason=f"model not downloaded: {'/'.join(parts)}"
    )


def ort_cpu_reference(path: str, feed: dict, output_names=None):
    """Run the ONNX reference implementation (ORT CPU EP)."""
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = output_names or [sess.get_outputs()[0].name]
    return sess.run(outs, feed)[0]


def make_feed(path: str, seed: int = 7) -> dict:
    """Deterministic random feed matching a model's declared inputs."""
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    feed = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dt = np.float64 if "double" in inp.type else np.float32
        feed[inp.name] = rng.normal(0, 0.4, shape).astype(dt).clip(-1, 1)
    return feed


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
