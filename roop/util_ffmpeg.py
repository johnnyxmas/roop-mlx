import hashlib
import json
import os
import subprocess
import tempfile
import roop.globals
import roop.utilities as util

from typing import List

_videotoolbox_available = None

# What an HTML5 <video> element can be relied on to play.  HEVC is
# deliberately excluded: Safari plays it, Chrome and Firefox do not do so
# reliably, and the preview should work in whichever browser is used.
_BROWSER_CONTAINERS = {".mp4", ".m4v", ".webm", ".ogv", ".ogg"}
_BROWSER_VIDEO_CODECS = {"h264", "vp8", "vp9", "av1"}
_BROWSER_AUDIO_CODECS = {"aac", "mp3", "opus", "vorbis"}


def is_videotoolbox_codec(codec: str) -> bool:
    return codec is not None and "videotoolbox" in codec


def detect_videotoolbox() -> bool:
    """Probe whether VideoToolbox hardware encoding is available.

    On Apple Silicon (and recent Intel Macs) ffmpeg exposes
    ``h264_videotoolbox`` and ``hevc_videotoolbox``.  We dry-run a
    0.1 s synth source so the probe matches reality, not just the
    codec list.
    """
    global _videotoolbox_available
    if _videotoolbox_available is not None:
        return _videotoolbox_available
    try:
        subprocess.check_output(
            [
                "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                "-c:v", "h264_videotoolbox",
                "-f", "null", "-",
            ],
            stderr=subprocess.STDOUT,
        )
        _videotoolbox_available = True
        print("VideoToolbox hardware encoding is available")
    except Exception:
        _videotoolbox_available = False
        print("VideoToolbox hardware encoding is NOT available, will use software encoding")
    return _videotoolbox_available


_fps_mode_supported = None


def detect_fps_mode() -> bool:
    """Probe whether this ffmpeg accepts ``-fps_mode``.

    ffmpeg 9 removed the ``-vsync`` alias outright ("Unrecognized option
    'vsync'"), which breaks frame extraction with an argument-parse error
    before any decoding happens.  ``-fps_mode`` has been the supported
    spelling since ffmpeg 5.0.  Probed with a dry run, like
    ``detect_videotoolbox``, so the answer matches the binary on PATH.
    """
    global _fps_mode_supported
    if _fps_mode_supported is not None:
        return _fps_mode_supported
    try:
        subprocess.check_output(
            [
                "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.04",
                "-fps_mode", "passthrough",
                "-f", "null", "-",
            ],
            stderr=subprocess.STDOUT,
        )
        _fps_mode_supported = True
    except Exception:
        _fps_mode_supported = False
    return _fps_mode_supported


def get_fps_mode_args() -> List[str]:
    """Args for "one output frame per input frame, no dupes or drops"."""
    if detect_fps_mode():
        return ["-fps_mode", "passthrough"]
    return ["-vsync", "0"]


def get_quality_args(codec: str, quality: int) -> List[str]:
    """Return the correct quality arguments for the given codec.

    VideoToolbox uses ``-q:v`` (numeric, higher is better) while
    libx264/libx265 use ``-crf``.  We keep the same numeric value to
    avoid breaking existing config files.
    """
    if is_videotoolbox_codec(codec):
        return ["-q:v", str(quality)]
    return ["-crf", str(quality)]


def get_software_fallback(codec: str) -> str:
    """Return the CPU software fallback for a VideoToolbox codec."""
    if codec == "h264_videotoolbox":
        return "libx264"
    if codec == "hevc_videotoolbox":
        return "libx265"
    return codec


def resolve_codec(codec: str) -> str:
    """Resolve 'auto' codec or validate VideoToolbox availability, falling back to software."""
    if codec == "auto":
        if detect_videotoolbox():
            return "h264_videotoolbox"
        return "libx264"
    if is_videotoolbox_codec(codec) and not detect_videotoolbox():
        fallback = get_software_fallback(codec)
        print(f"VideoToolbox not available, falling back from {codec} to {fallback}")
        return fallback
    return codec


def run_ffmpeg(args: List[str]) -> bool:
    commands = [
        "ffmpeg",
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-y",
        "-loglevel",
        roop.globals.log_level,
    ]
    commands.extend(args)
    print("Running ffmpeg")
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        print("Running ffmpeg failed! Commandline:")
        print(" ".join(commands))
    return False


def _probe_duration_seconds(path: str):
    """Total duration of *path* in seconds via ffprobe, or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", path,
            ],
            stderr=subprocess.DEVNULL, timeout=15,
        )
        return float(out.decode().strip())
    except Exception:
        return None


def _parse_ffmpeg_time(value: str):
    """Parse an ffmpeg out_time value (HH:MM:SS.uuuuuu) into seconds."""
    try:
        hh, mm, ss = value.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return None


def scale_video(
    input_path: str,
    output_path: str,
    target_height: int = None,
    target_fps: float = None,
    on_progress=None,
) -> bool:
    """Re-encode *input_path*, optionally downscaling to *target_height* px tall
    (aspect preserved, even width) and/or changing the frame rate to *target_fps*.

    Uses h264_videotoolbox when available (Apple Silicon hardware encoder),
    falling back to libx264 via resolve_codec("auto"). Audio is re-encoded to
    AAC. Calls on_progress(0.0-1.0) if provided. Returns True on success.
    """
    codec = resolve_codec("auto")
    quality = roop.globals.video_quality if roop.globals.video_quality else 18
    duration = _probe_duration_seconds(input_path)

    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-loglevel", "error", "-i", input_path,
    ]
    if target_height:
        cmd.extend(["-vf", f"scale=-2:{target_height}"])
    if target_fps:
        cmd.extend(["-r", str(target_fps)])
    cmd.extend(["-c:v", codec])
    cmd.extend(get_quality_args(codec, quality))
    cmd.extend(["-c:a", "aac", "-progress", "pipe:1", "-nostats", output_path])

    print("Running ffmpeg (re-encode)")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, bufsize=1,
        )
    except Exception:
        print("Failed to start ffmpeg. Commandline:")
        print(" ".join(cmd))
        return False

    try:
        for line in proc.stdout:
            line = line.strip()
            if on_progress and duration and line.startswith("out_time="):
                secs = _parse_ffmpeg_time(line.split("=", 1)[1])
                if secs is not None:
                    on_progress(max(0.0, min(0.99, secs / duration)))
            elif line == "progress=end" and on_progress:
                on_progress(1.0)
    finally:
        proc.wait()

    ok = proc.returncode == 0 and os.path.exists(output_path)
    if not ok:
        print("ffmpeg re-encode failed. Commandline:")
        print(" ".join(cmd))
    return ok


def probe_media_codecs(path: str):
    """Return ``(video_codec, audio_codec)`` for *path*.

    Either entry is ``None`` when that stream is absent or ffprobe fails.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,codec_name",
                "-of", "json", path,
            ],
            stderr=subprocess.DEVNULL, timeout=15,
        )
        streams = json.loads(out.decode()).get("streams", [])
    except Exception:
        return None, None

    video = audio = None
    for stream in streams:
        kind = stream.get("codec_type")
        if kind == "video" and video is None:
            video = stream.get("codec_name")
        elif kind == "audio" and audio is None:
            audio = stream.get("codec_name")
    return video, audio


def is_browser_compatible_video(path: str) -> bool:
    """True when *path* should play directly in an HTML5 video element.

    Checks the container and both stream codecs -- a swap output can end
    up unplayable either from the codec settings (libx265, mkv/avi) or
    from ``restore_audio`` copying an exotic audio track off the source.

    Returns True when probing fails, so an unreadable file never triggers
    a pointless re-encode.
    """
    if os.path.splitext(path)[1].lower() not in _BROWSER_CONTAINERS:
        return False

    video, audio = probe_media_codecs(path)
    if video is None:
        return True
    if video not in _BROWSER_VIDEO_CODECS:
        return False
    if audio is not None and audio not in _BROWSER_AUDIO_CODECS:
        return False
    return True


def create_browser_preview(video_path: str, dest_dir: str = None, on_progress=None):
    """Re-encode *video_path* into a browser-playable H.264/AAC MP4.

    The preview lands in Gradio's temp directory (so it is served to the
    UI and cleared by "Clean temp folder") and is reused while it stays
    newer than its source.  Returns the preview path, or None on failure.
    """
    if dest_dir is None:
        dest_dir = (
            os.environ.get("GRADIO_TEMP_DIR")
            or os.environ.get("TEMP")
            or tempfile.gettempdir()
        )
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError:
        return None

    source = os.path.abspath(video_path)
    stem = os.path.splitext(os.path.basename(source))[0]
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    preview = os.path.join(dest_dir, f"{stem}_{digest}_preview.mp4")

    try:
        if (os.path.exists(preview)
                and os.path.getmtime(preview) >= os.path.getmtime(source)):
            return preview
    except OSError:
        pass

    if scale_video(source, preview, on_progress=on_progress):
        return preview
    return None


def cut_video(
    original_video: str,
    cut_video: str,
    start_frame: int,
    end_frame: int,
    reencode: bool,
):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    if reencode:
        codec = resolve_codec(roop.globals.video_encoder)
        args = [
            "-ss",
            format(start_time, ".2f"),
            "-i",
            original_video,
            "-c:v",
            codec,
        ]
        args.extend(get_quality_args(codec, roop.globals.video_quality))
        args.extend(
            [
                "-c:a",
                "aac",
                "-frames:v",
                str(num_frames),
                cut_video,
            ]
        )
        run_ffmpeg(args)
    else:
        run_ffmpeg(
            [
                "-ss",
                format(start_time, ".2f"),
                "-i",
                original_video,
                "-frames:v",
                str(num_frames),
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                cut_video,
            ]
        )


def join_videos(videos: List[str], dest_filename: str, simple: bool):
    if simple:
        txtfilename = util.resolve_relative_path("../temp")
        txtfilename = os.path.join(txtfilename, "joinvids.txt")
        with open(txtfilename, "w", encoding="utf-8") as f:
            for v in videos:
                v = v.replace("\\", "/")
                f.write(f"file {v}\n")
        commands = [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            f"{txtfilename}",
            "-vcodec",
            "copy",
            f"{dest_filename}",
        ]
        run_ffmpeg(commands)

    else:
        inputs = []
        filter = ""
        for i, v in enumerate(videos):
            inputs.append("-i")
            inputs.append(v)
            filter += f"[{i}:v:0][{i}:a:0]"
        run_ffmpeg(
            [
                " ".join(inputs),
                "-filter_complex",
                f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"',
                "-map",
                '"[outv]"',
                "-map",
                '"[outa]"',
                dest_filename,
            ]
        )

        #     filter += f'[{i}:v:0][{i}:a:0]'
        # run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])


def extract_frames(
    target_path: str, trim_frame_start, trim_frame_end, fps: float
) -> bool:
    util.create_temp(target_path)
    temp_directory_path = util.get_temp_directory_path(target_path)
    commands = [
        "-i",
        target_path,
        "-q:v",
        "1",
        "-pix_fmt",
        "rgb24",
    ]
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend(
            [
                "-vf",
                "trim=start_frame="
                + str(trim_frame_start)
                + ":end_frame="
                + str(trim_frame_end)
                + ",fps="
                + str(fps),
            ]
        )
    commands.extend(
        [
            *get_fps_mode_args(),
            os.path.join(
                temp_directory_path, "%06d." + roop.globals.CFG.output_image_format
            ),
        ]
    )
    return run_ffmpeg(commands)


def create_video(
    target_path: str,
    dest_filename: str,
    fps: float = 24.0,
    temp_directory_path: str = None,
) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    codec = resolve_codec(roop.globals.video_encoder)
    args = [
        "-r",
        str(fps),
        "-i",
        os.path.join(
            temp_directory_path, f"%06d.{roop.globals.CFG.output_image_format}"
        ),
        "-c:v",
        codec,
    ]
    args.extend(get_quality_args(codec, roop.globals.video_quality))
    args.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "colorspace=bt709:iall=bt601-6-625:fast=1",
            "-y",
            dest_filename,
        ]
    )
    if not run_ffmpeg(args) and is_videotoolbox_codec(codec):
        fallback = get_software_fallback(codec)
        print(f"VideoToolbox encoding failed, retrying with {fallback}")
        args[args.index(codec)] = fallback
        # Swap VideoToolbox quality args for libx264/libx265 -crf style
        if "-q:v" in args:
            q_idx = args.index("-q:v")
            quality_val = args[q_idx + 1]
            del args[q_idx:q_idx + 2]
            args[q_idx:q_idx] = ["-crf", quality_val]
        run_ffmpeg(args)
    return dest_filename


def create_gif_from_video(video_path: str, gif_path):
    from roop.capturer import get_video_frame, release_video

    fps = util.detect_fps(video_path)
    frame = get_video_frame(video_path)
    release_video()

    scalex = frame.shape[0]
    scaley = frame.shape[1]

    if scalex >= scaley:
        scaley = -1
    else:
        scalex = -1

    run_ffmpeg(
        [
            "-i",
            video_path,
            "-vf",
            f"fps={fps},scale={int(scalex)}:{int(scaley)}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            gif_path,
        ]
    )


def create_video_from_gif(gif_path: str, output_path):
    filter = """scale='trunc(in_w/2)*2':'trunc(in_h/2)*2',format=yuv420p,fps=10"""
    run_ffmpeg(
        [
            "-i",
            gif_path,
            "-vf",
            f'"{filter}"',
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ]
    )


def repair_video(original_video: str, final_video: str):
    run_ffmpeg(
        [
            "-i",
            original_video,
            "-movflags",
            "faststart",
            "-acodec",
            "copy",
            "-vcodec",
            "copy",
            final_video,
        ]
    )


def restore_audio(
    intermediate_video: str,
    original_video: str,
    trim_frame_start,
    trim_frame_end,
    final_video: str,
) -> None:
    fps = util.detect_fps(original_video)
    commands = ["-i", intermediate_video]
    if trim_frame_start is None and trim_frame_end is None:
        commands.extend(["-c:a", "copy"])
    else:
        # if trim_frame_start is not None:
        # 	start_time = trim_frame_start / fps
        # 	commands.extend([ '-ss', format(start_time, ".2f")])
        # else:
        # 	commands.extend([ '-ss', '0' ])
        # if trim_frame_end is not None:
        # 	end_time = trim_frame_end / fps
        # 	commands.extend([ '-to', format(end_time, ".2f")])
        # commands.extend([ '-c:a', 'aac' ])
        if trim_frame_start is not None:
            start_time = trim_frame_start / fps
            commands.extend(["-ss", format(start_time, ".2f")])
        else:
            commands.extend(["-ss", "0"])
        if trim_frame_end is not None:
            end_time = trim_frame_end / fps
            commands.extend(["-to", format(end_time, ".2f")])
        commands.extend(["-i", original_video, "-c", "copy"])

    commands.extend(["-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_video])
    run_ffmpeg(commands)
