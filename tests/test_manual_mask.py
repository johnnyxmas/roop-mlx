"""Manual-mask toggle: the hidden editor passes None before first use."""
import numpy as np
import pytest


@pytest.fixture()
def tab():
    from ui.tabs import faceswap_tab
    faceswap_tab.manual_masking = False
    yield faceswap_tab
    faceswap_tab.manual_masking = False


def test_toggle_on_with_untouched_editor(tab):
    """Regression: mask is None until the editor has held a value, and
    on_toggle_masking subscripted it (TypeError)."""
    preview = np.zeros((48, 64, 3), dtype=np.uint8)
    img, editor = tab.on_toggle_masking(preview, None)
    assert tab.manual_masking is True
    assert img.visible is False and editor.visible is True
    tab.on_toggle_masking(preview, None)          # and back off
    assert tab.manual_masking is False


def test_toggle_without_preview_stays_off(tab):
    img, editor = tab.on_toggle_masking(None, None)
    assert tab.manual_masking is False
    assert img.visible is True and editor.visible is False


@pytest.fixture()
def one_face(tab):
    import roop.globals
    from roop.FaceSet import FaceSet
    from roop.face_detector import FaceCompat

    saved = roop.globals.INPUT_FACESETS
    fs = FaceSet()
    fs.faces.append(FaceCompat())
    roop.globals.INPUT_FACESETS = [fs]
    tab.SELECTED_INPUT_FACE_INDEX = 0
    yield fs.faces[0]
    roop.globals.INPUT_FACESETS = saved


def test_mask_sliders_update_tuple_offsets(tab, one_face):
    """Regression: faces store mask_offsets as a tuple, and set_mask_offset
    assigned into it in place (TypeError on any mask slider)."""
    tab.on_mask_blur_changed(35)
    tab.on_mask_top_changed(0.2)
    assert one_face.mask_offsets == (0.2, 0, 0, 0, 1, 35)


def test_mask_offsets_clamp_opposite_edges(tab, one_face):
    tab.on_mask_left_changed(0.7)
    tab.on_mask_right_changed(0.5)          # left + right > 0.99
    assert one_face.mask_offsets[2:4] == (0.99, 0.0)
