"""Mask_Clip2Seg: text-prompted masking through the ONNX CLIPSeg export."""

import numpy as np

from conftest import requires_model


@requires_model("CLIP", "clipseg_rd64_refined.onnx")
def test_clip2seg_mask_contract(rng):
    """Run() must return a [0,1] float mask at the model's native size and
    respond to the prompt (different prompts -> different masks)."""
    from roop.processors.Mask_Clip2Seg import Mask_Clip2Seg

    p = Mask_Clip2Seg()
    p.Initialize({})
    try:
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8).copy()
        mask_a = p.Run(frame, "face")
        assert mask_a.shape == (352, 352)
        assert mask_a.dtype == np.float32
        assert 0.0 <= mask_a.min() and mask_a.max() <= 1.0

        # prompt must condition the model: raw logits differ between prompts
        # (on noise input both thresholded masks are legitimately all-zero)
        import cv2
        from roop.processors.Mask_Clip2Seg import _INPUT_SIZE, _MEAN, _STD
        small = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE))
        rgb = (small[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
        pv = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])
        la = p._predict(pv, "face")
        lb = p._predict(pv, "background")
        assert np.abs(la - lb).max() > 0.1, "prompt has no effect on logits"

        # empty prompt passes the frame through untouched
        assert p.Run(frame, "") is frame
    finally:
        p.Release()


@requires_model("CLIP", "clipseg_rd64_refined.onnx")
def test_clip2seg_tokenizer_roundtrip():
    from roop.processors.Mask_Clip2Seg import Mask_Clip2Seg, _CONTEXT_LENGTH

    p = Mask_Clip2Seg()
    p.Initialize({})
    try:
        ids, mask = p._tokenize("short prompt")
        assert ids.shape == mask.shape == (1, _CONTEXT_LENGTH)
        assert ids[0, 0] == p._sot
        n = int(mask.sum())
        assert ids[0, n - 1] == p._eot and ids[0, n:].sum() == 0
    finally:
        p.Release()
