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
