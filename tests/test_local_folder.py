"""Add-local-folder listing: what reaches the Gradio file component."""
import os

from roop.utilities import get_local_files_from_folder


def test_hidden_files_are_skipped(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / "._clip.mp4").write_bytes(b"x")
    (tmp_path / "nested").mkdir()

    files = get_local_files_from_folder(str(tmp_path))

    assert [os.path.basename(f) for f in files] == ["clip.mp4"]


def test_missing_folder_returns_none(tmp_path):
    assert get_local_files_from_folder(str(tmp_path / "absent")) is None
