"""Browser-compatibility detection and the opt-in preview re-encode."""

import os
import shutil
import subprocess

import pytest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _encode(path, vcodec="libx264", acodec=None, extra=()):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
    ]
    if acodec:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:a", acodec]
    cmd += ["-c:v", vcodec, "-pix_fmt", "yuv420p", *extra, "-y", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


@pytest.fixture(scope="module")
def cfg():
    """Minimal globals so util_ffmpeg can run outside the app."""
    import roop.globals
    from settings import Settings

    if roop.globals.CFG is None:
        roop.globals.CFG = Settings("config.yaml")
    roop.globals.video_quality = roop.globals.video_quality or 70
    roop.globals.log_level = "error"
    return roop.globals.CFG


@needs_ffmpeg
@pytest.mark.parametrize("name,vcodec,acodec,compatible", [
    ("ok.mp4", "libx264", "aac", True),
    ("hevc.mp4", "libx265", None, False),
    ("badcontainer.mkv", "libx264", None, False),
    ("badaudio.mp4", "libx264", "ac3", False),
])
def test_browser_compatibility_detection(tmp_path, cfg, name, vcodec, acodec, compatible):
    import roop.util_ffmpeg as ffmpeg

    path = _encode(tmp_path / name, vcodec, acodec)
    assert ffmpeg.is_browser_compatible_video(path) is compatible


def test_unprobeable_file_is_left_alone(tmp_path, cfg):
    """A failed probe must not trigger a pointless re-encode."""
    import roop.util_ffmpeg as ffmpeg

    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"not a video")
    assert ffmpeg.is_browser_compatible_video(str(bogus)) is True


@needs_ffmpeg
def test_preview_reencode_is_playable_and_cached(tmp_path, cfg):
    import roop.util_ffmpeg as ffmpeg

    source = _encode(tmp_path / "hevc.mp4", "libx265")
    dest = tmp_path / "previews"

    preview = ffmpeg.create_browser_preview(source, dest_dir=str(dest))
    assert preview is not None and os.path.exists(preview)
    assert ffmpeg.is_browser_compatible_video(preview)
    # the saved output must never be touched
    assert ffmpeg.probe_media_codecs(source)[0] == "hevc"

    again = ffmpeg.create_browser_preview(source, dest_dir=str(dest))
    assert again == preview
    assert len(list(dest.iterdir())) == 1


@needs_ffmpeg
def test_preview_reencode_transcodes_audio(tmp_path, cfg):
    import roop.util_ffmpeg as ffmpeg

    source = _encode(tmp_path / "ac3.mp4", "libx264", "ac3")
    preview = ffmpeg.create_browser_preview(source, dest_dir=str(tmp_path / "p"))
    assert ffmpeg.probe_media_codecs(preview) == ("h264", "aac")


def test_failed_reencode_returns_none(tmp_path, cfg):
    import roop.util_ffmpeg as ffmpeg

    bogus = tmp_path / "bogus.mkv"
    bogus.write_bytes(b"not a video")
    assert ffmpeg.create_browser_preview(str(bogus), dest_dir=str(tmp_path / "p")) is None


@needs_ffmpeg
def test_reencode_only_runs_when_setting_enabled(tmp_path, cfg, monkeypatch):
    """The whole point of the setting: nothing re-encodes unprompted."""
    import gradio as gr
    import roop.util_ffmpeg as ffmpeg
    from ui.tabs import faceswap_tab

    monkeypatch.setattr(gr, "Info", lambda *a, **k: None)
    monkeypatch.setattr(gr, "Warning", lambda *a, **k: None)

    calls = []
    real = ffmpeg.create_browser_preview
    monkeypatch.setattr(
        ffmpeg, "create_browser_preview",
        lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1],
    )
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gtmp"))

    playable = _encode(tmp_path / "ok.mp4", "libx264", "aac")
    unplayable = _encode(tmp_path / "hevc.mp4", "libx265")

    monkeypatch.setattr(cfg, "output_show_video", True)
    monkeypatch.setattr(cfg, "output_reencode_preview", False)

    assert faceswap_tab._browser_playable_path(playable) == playable
    assert faceswap_tab._browser_playable_path(unplayable) is None
    assert calls == [], "re-encoded despite the setting being disabled"

    monkeypatch.setattr(cfg, "output_reencode_preview", True)
    preview = faceswap_tab._browser_playable_path(unplayable)
    assert preview is not None and preview != unplayable
    assert calls == [unplayable]


def test_setting_defaults_off_and_roundtrips(tmp_path):
    """Existing configs (written before this setting) must default to off."""
    from settings import Settings

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("output_show_video: true\n")

    cfg = Settings(str(cfg_path))
    assert cfg.output_reencode_preview is False

    cfg.output_reencode_preview = True
    cfg.save()
    assert Settings(str(cfg_path)).output_reencode_preview is True
