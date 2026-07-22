"""Pure-numpy ProcessMgr math: pixel boost, batched prepare/normalize,
and the ROI paste_upscale against the original full-frame implementation."""

import types

import cv2
import numpy as np
import pytest


@pytest.fixture()
def mgr():
    from roop.ProcessMgr import ProcessMgr

    m = ProcessMgr(None)
    m.options = types.SimpleNamespace(
        blend_ratio=0.65,
        show_face_area_overlay=False,
        swap_modelname="InSwapper 128",
    )
    return m


def test_pixel_boost_roundtrip(mgr):
    """implode -> explode with no swap must reproduce the input exactly."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    slices = mgr.implode_pixel_boost(img, 128, 4)
    assert slices.shape == (16, 128, 128, 3)
    out = mgr.explode_pixel_boost(list(slices), 128, 4, 512)
    assert np.array_equal(out, img)


@pytest.mark.parametrize("model_type", ["inswapper", "ghost"])
def test_prepare_normalize_roundtrip(mgr, model_type):
    """normalize must invert prepare (up to uint8 rounding)."""
    mgr._get_swap_type = lambda: model_type
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    blob = mgr.prepare_crop_frame(frame)
    assert blob.shape == (1, 3, 128, 128) and blob.dtype == np.float32
    back = mgr.normalize_swap_frame(blob)
    assert np.abs(back - frame).max() <= 1


def _paste_upscale_fullframe(self, fake_face, upsk_face, M, target_img,
                             scale_factor, mask_offsets):
    """Original full-frame implementation, kept as the reference."""
    M_scale = M * scale_factor
    IM = cv2.invertAffineTransform(M_scale)
    img_matte = np.zeros((upsk_face.shape[0], upsk_face.shape[1]), dtype=np.uint8)
    w, h = img_matte.shape[1], img_matte.shape[0]
    top = int(mask_offsets[0] * h)
    bottom = int(h - (mask_offsets[1] * h))
    left = int(mask_offsets[2] * w)
    right = int(w - (mask_offsets[3] * w))
    img_matte[top:bottom, left:right] = 255
    img_matte = cv2.warpAffine(img_matte, IM,
                               (target_img.shape[1], target_img.shape[0]),
                               flags=cv2.INTER_NEAREST, borderValue=0.0)
    img_matte[:1, :] = img_matte[-1:, :] = img_matte[:, :1] = img_matte[:, -1:] = 0
    img_matte = self.blur_area(img_matte, mask_offsets[4], mask_offsets[5])
    img_matte = (img_matte.astype(np.float32) / 255)[..., None]
    paste_face = cv2.warpAffine(upsk_face, IM,
                                (target_img.shape[1], target_img.shape[0]),
                                borderMode=cv2.BORDER_REPLICATE)
    if upsk_face is not fake_face:
        fake_face = cv2.warpAffine(fake_face, IM,
                                   (target_img.shape[1], target_img.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        paste_face = cv2.addWeighted(paste_face, self.options.blend_ratio,
                                     fake_face, 1.0 - self.options.blend_ratio, 0)
    paste_face = img_matte * paste_face
    paste_face = paste_face + (1 - img_matte) * target_img.astype(np.float32)
    return paste_face.astype(np.uint8)


MASK_OFFSETS = (0.0, 0.0, 0.0, 0.0, 1, 20)


@pytest.mark.parametrize("case,M", [
    ("centered", cv2.getRotationMatrix2D((640, 360), 10, 512 / 300)),
    ("edge-clipped", cv2.getRotationMatrix2D((40, 40), -25, 512 / 200)),
])
@pytest.mark.parametrize("enhanced", [False, True])
def test_paste_upscale_roi_matches_fullframe(mgr, case, M, enhanced):
    rng = np.random.default_rng(4)
    target = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    crop = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    upsk = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8) if enhanced else crop

    M = M.astype(np.float64)
    old = _paste_upscale_fullframe(mgr, crop, upsk, M, target, 1, MASK_OFFSETS)
    new = mgr.paste_upscale(crop, upsk, M, target, 1, MASK_OFFSETS)

    diff = np.abs(old.astype(np.int32) - new.astype(np.int32))
    frac_diff = (diff.max(axis=-1) > 0).mean()
    # Differences are confined to the feather fringe (ROI-border Gaussian
    # extrapolation); measured: 0 for edge-clipped, ~0.25% / max 8 centered.
    assert diff.max() <= 12, f"{case}: max channel delta {diff.max()}"
    assert frac_diff < 0.01, f"{case}: {frac_diff * 100:.2f}% pixels differ"


def test_paste_upscale_face_off_frame_returns_frame(mgr):
    rng = np.random.default_rng(5)
    target = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    crop = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    M = np.array([[1.7, 0.0, -4000.0], [0.0, 1.7, -4000.0]], dtype=np.float64)
    out = mgr.paste_upscale(crop, crop, M, target, 1, MASK_OFFSETS)
    assert np.array_equal(out, target)
