"""Identity weighting (FaceFusion's face_swapper_weight, ported 2026-09-16).

The control is "target-ward <-> source-ward", not a blend amount: the 0..1 UI
range maps to +0.35..-0.35, so 0.5 is neutral and 1.0 extrapolates past the
source.  The default must be a provable no-op — that is what makes this safe
to land without re-validating every swap model.
"""

import numpy as np
import pytest

from conftest import requires_model

from roop.face_util import balance_source_embedding


def _unit(v):
    return np.asarray(v, dtype=np.float32) / np.linalg.norm(v)


SRC = _unit([1.0, 0.0, 0.0, 0.0]).reshape(1, -1)
TGT = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)


def test_default_weight_is_bit_identical():
    out = balance_source_embedding(SRC, TGT, 0.5)
    assert np.array_equal(out, SRC)


def test_lower_weight_moves_toward_target():
    out = balance_source_embedding(SRC, TGT, 0.0)
    # 0.35 of a unit target mixed in, source scaled by 0.65.
    assert out[0, 0] == pytest.approx(0.65)
    assert out[0, 1] == pytest.approx(0.35)


def test_upper_weight_extrapolates_away_from_target():
    out = balance_source_embedding(SRC, TGT, 1.0)
    assert out[0, 0] == pytest.approx(1.35)
    assert out[0, 1] == pytest.approx(-0.35)


def test_target_is_normalised_before_blending():
    """Target magnitude must not leak into the result (upstream normalises it)."""
    small = balance_source_embedding(SRC, TGT, 0.0)
    large = balance_source_embedding(SRC, TGT * 1000.0, 0.0)
    assert np.allclose(small, large)


def test_result_is_not_renormalised():
    """Upstream leaves the blend un-normalised; don't silently 'fix' it."""
    out = balance_source_embedding(SRC, TGT, 0.0)
    assert np.linalg.norm(out) != pytest.approx(1.0)


@pytest.mark.parametrize("missing", ["target", "source"])
def test_missing_embedding_falls_back_to_source(missing):
    """The tracker reuses faces across frames and may carry embedding=None."""
    if missing == "target":
        assert np.array_equal(balance_source_embedding(SRC, None, 0.0), SRC)
    else:
        assert balance_source_embedding(None, TGT, 0.0) is None


def test_process_options_defaults_to_the_global():
    import roop.globals
    from roop.ProcessOptions import ProcessOptions

    previous = roop.globals.swapper_weight
    try:
        roop.globals.swapper_weight = 0.25
        opts = ProcessOptions("InSwapper 128", {}, 0.65, 0.65, "first", 0,
                              "", None, 1, 128, False, False)
        assert opts.swapper_weight == 0.25
        explicit = ProcessOptions("InSwapper 128", {}, 0.65, 0.65, "first", 0,
                                  "", None, 1, 128, False, False,
                                  swapper_weight=0.8)
        assert explicit.swapper_weight == 0.8
    finally:
        roop.globals.swapper_weight = previous


class _FakeFace:
    """Stand-in for the detector's Face object (see face_detector.py)."""

    def __init__(self, normed_embedding=None, embedding=None):
        self.normed_embedding = normed_embedding
        self.embedding = embedding


@requires_model("hyperswap_1b_256.onnx")
def test_weight_reaches_the_model_end_to_end():
    """plugin_options plumbing: the slider must actually change the swap."""
    from roop.processors.FaceSwapFaceFusion import FaceSwapFaceFusion

    rng = np.random.default_rng(5)
    source = rng.normal(size=512).astype(np.float32)
    src = _FakeFace(normed_embedding=source / np.linalg.norm(source))
    tgt = _FakeFace(embedding=rng.normal(size=512).astype(np.float32))
    frame = rng.uniform(-1, 1, (1, 3, 256, 256)).astype(np.float32)

    def run(weight):
        proc = FaceSwapFaceFusion()
        proc.Initialize({"modelname": "HyperSwap 1B", "swapper_weight": weight})
        out = proc.Run(src, tgt, frame)
        proc.Release()
        return np.asarray(out, dtype=np.float64)

    neutral = run(0.5)
    assert np.array_equal(neutral, run(0.5)), "swap is not deterministic"
    assert not np.allclose(neutral, run(0.2)), "lower weight had no effect"
    assert not np.allclose(neutral, run(0.9)), "higher weight had no effect"
