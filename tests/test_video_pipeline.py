"""Video I/O: AVFoundation reader fallback and BT.709 color round-trip.

Requires ffmpeg on PATH; generates tiny synthetic clips in tmp_path.
"""

import shutil
import subprocess

import cv2
import numpy as np
import pytest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _make_clip(path, size="640x360", codec="libx264", frames=30, flagged=True):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"gradients=size={size}:rate=30:duration=1:speed=0",
        "-frames:v", str(frames), "-c:v", codec, "-pix_fmt", "yuv420p",
    ]
    if flagged:
        cmd += ["-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-color_range", "tv"]
    cmd += ["-y", str(path)]
    subprocess.run(cmd, check=True)


def _truth_frame(video_path, png_path):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
         "-frames:v", "1", "-y", str(png_path)],
        check=True,
    )
    return cv2.imread(str(png_path))


@needs_ffmpeg
def test_reader_matches_ffmpeg_truth_on_flagged_bt709(tmp_path):
    """AVFoundation honors colorspace flags; cv2's FFmpeg backend does not."""
    from roop.capturer import open_video_capture

    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    truth = _truth_frame(clip, tmp_path / "truth.png")

    cap = open_video_capture(str(clip))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == truth.shape
    d = np.abs(frame.astype(int) - truth.astype(int))
    assert d.mean() < 2.0, f"decoded colors drifted from truth: mean {d.mean():.2f}"


@needs_ffmpeg
def test_reader_falls_back_for_gif(tmp_path):
    from roop.capturer import open_video_capture

    clip = tmp_path / "clip.mp4"
    _make_clip(clip, size="320x180", frames=10)
    gif = tmp_path / "clip.gif"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(clip),
         "-y", str(gif)],
        check=True,
    )
    cap = open_video_capture(str(gif))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None and frame.size > 0


@needs_ffmpeg
def test_writer_roundtrip_preserves_color_and_tags(tmp_path):
    from roop.capturer import open_video_capture
    from roop.ffmpeg_writer import FFMPEG_VideoWriter

    src = tmp_path / "src.mp4"
    _make_clip(src)
    truth = _truth_frame(src, tmp_path / "src.png")

    out = tmp_path / "out.mp4"
    cap = open_video_capture(str(src))
    writer = FFMPEG_VideoWriter(str(out), (640, 360), 30.0, codec="auto", crf=90)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write_frame(frame)
        n += 1
    cap.release()
    writer.close()
    assert n == 30

    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-select_streams", "v",
         "-show_entries", "stream=color_space,color_primaries,color_transfer,color_range",
         "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "color_space=bt709" in probe
    assert "color_primaries=bt709" in probe
    assert "color_range=tv" in probe

    rt = _truth_frame(out, tmp_path / "rt.png")
    d = np.abs(truth.astype(int) - rt.astype(int))
    # codec quantization only; a colorspace-matrix mismatch would be mean ~7
    assert d.mean() < 3.0, f"round-trip color shift: mean {d.mean():.2f}"


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["bgra", "gray", "too_wide", "float"])
def test_writer_coerces_malformed_frames_without_shear(tmp_path, kind):
    """A frame whose shape/channels/dtype differ from the declared size would
    desync the raw bgr24 stride and shear the output; the writer must coerce
    it back to an exact (H, W, 3) uint8 buffer instead."""
    from roop.ffmpeg_writer import FFMPEG_VideoWriter

    w, h = 852, 480  # non-16-aligned width, the classic shear trigger
    # Horizontal gradient: shear or a wrong stride shows as a non-monotonic row.
    ramp = np.repeat(
        np.linspace(0, 255, w, dtype=np.uint8)[None, :], h, axis=0
    )[..., None].repeat(3, axis=2)

    if kind == "bgra":
        frame = np.dstack([ramp, np.full((h, w, 1), 255, np.uint8)])
    elif kind == "gray":
        frame = cv2.cvtColor(ramp, cv2.COLOR_BGR2GRAY)
    elif kind == "too_wide":
        frame = np.repeat(
            np.linspace(0, 255, w + 1, dtype=np.uint8)[None, :], h, axis=0
        )[..., None].repeat(3, axis=2)
    else:  # float
        frame = ramp.astype(np.float32)

    out = tmp_path / "out.mp4"
    writer = FFMPEG_VideoWriter(str(out), (w, h), 10.0, codec="libx264", crf=95)
    for _ in range(3):
        writer.write_frame(frame.copy())
    writer.close()

    cap = cv2.VideoCapture(str(out))
    ok, decoded = cap.read()
    cap.release()
    assert ok and decoded.shape == (h, w, 3)
    # A sheared frame wraps content and breaks the left-to-right ramp.
    col_means = decoded.mean(axis=(0, 2))
    assert np.all(np.diff(col_means) >= -3), "output frame is sheared"


@needs_ffmpeg
def test_fps_mode_args_are_accepted_by_this_ffmpeg():
    """ffmpeg 9 removed ``-vsync``; the extract path must use what is installed.

    Asserts against the real binary on PATH rather than a version string, so
    the test keeps passing on ffmpeg 5-8 (where both spellings work) and fails
    loudly if the chosen flag is ever rejected again.
    """
    from roop import util_ffmpeg

    util_ffmpeg._fps_mode_supported = None
    args = util_ffmpeg.get_fps_mode_args()

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.04", *args,
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Unrecognized option" not in proc.stderr


@needs_ffmpeg
def test_extract_frames_writes_frames(tmp_path, monkeypatch):
    """End-to-end guard on the keep-frames / Extras extraction path."""
    import roop.globals
    from roop import util_ffmpeg
    from roop import utilities as util

    class _CFG:
        output_image_format = "png"

    monkeypatch.setattr(roop.globals, "CFG", _CFG(), raising=False)

    clip = tmp_path / "clip.mp4"
    _make_clip(clip, size="64x64", frames=5)

    assert util_ffmpeg.extract_frames(str(clip), None, None, 30.0) is True
    assert len(util.get_temp_frame_paths(str(clip))) == 5
