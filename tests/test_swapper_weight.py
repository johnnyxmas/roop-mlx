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


@requires_model("alphaface_256.onnx")
def test_alphaface_source_choice_is_output_equivalent_and_keeps_the_weight_live():
    """Why this port feeds AlphaFace the unit-norm vector, not the raw one.

    Upstream uses source_face.embedding.  AlphaFace normalises its source
    internally, so both give the same swap -- but a +-0.35 identity shift is
    negligible against a raw ArcFace vector and meaningful against a unit-norm
    one, so the raw choice would silently disable the weight slider for this
    model alone.
    """
    import onnxruntime as ort

    from conftest import model_path

    ort.set_default_logger_severity(3)
    rng = np.random.default_rng(4)
    raw = (rng.normal(size=(1, 512)) * 20).astype(np.float32)
    unit = raw / np.linalg.norm(raw)
    target = (rng.normal(size=512) * 20).astype(np.float32)
    frame = rng.uniform(0, 1, (1, 3, 256, 256)).astype(np.float32)

    sess = ort.InferenceSession(
        model_path("alphaface_256.onnx"), providers=["CPUExecutionProvider"]
    )

    def swap(source):
        return sess.run(
            None, {"source": source.astype(np.float32), "target": frame}
        )[0].astype(np.float64)

    # Same swap either way: the model normalises the source itself.
    assert np.abs(swap(raw) - swap(unit)).max() < 1e-3

    def weight_effect(base):
        neutral = swap(balance_source_embedding(base, target, 0.5))
        shifted = swap(balance_source_embedding(base, target, 0.2))
        return np.abs(neutral - shifted).max()

    assert weight_effect(raw) < 0.01, "raw vector should barely move (upstream)"
    assert weight_effect(unit) > 0.05, "unit-norm vector must keep the weight live"


@requires_model("alphaface_256.onnx")
def test_alphaface_runs_through_the_processor():
    """Smoke test of the whole plugin path: routing, session, I/O shape."""
    from roop.processors.FaceSwapFaceFusion import FaceSwapFaceFusion

    rng = np.random.default_rng(23)
    emb = rng.normal(size=512).astype(np.float32)
    src = _FakeFace(normed_embedding=emb / np.linalg.norm(emb), embedding=emb)
    tgt = _FakeFace(embedding=rng.normal(size=512).astype(np.float32))
    frame = rng.uniform(0, 1, (1, 3, 256, 256)).astype(np.float32)

    proc = FaceSwapFaceFusion()
    proc.Initialize({"modelname": "AlphaFace 256", "swapper_weight": 0.5})
    try:
        assert proc.swap_type == "alphaface"
        out = np.asarray(proc.Run(src, tgt, frame))
        assert out.shape == (1, 3, 256, 256)
        # mean 0 / std 1 model: output is [0,1], not the +-1 HyperSwap range.
        assert -0.05 <= out.min() and out.max() <= 1.05
    finally:
        proc.Release()
