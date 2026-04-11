import os
import subprocess
import roop.globals
import roop.utilities as util

from typing import List, Any

_nvenc_available = None


def is_nvenc_codec(codec: str) -> bool:
    return codec is not None and "nvenc" in codec


def detect_nvenc() -> bool:
    """Probe whether NVENC encoding is available on this system."""
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available
    try:
        subprocess.check_output(
            [
                "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                "-c:v", "h264_nvenc",
                "-f", "null", "-",
            ],
            stderr=subprocess.STDOUT,
        )
        _nvenc_available = True
        print("NVENC hardware encoding is available")
    except Exception:
        _nvenc_available = False
        print("NVENC hardware encoding is NOT available, will use software encoding")
    return _nvenc_available


def get_quality_args(codec: str, quality: int) -> List[str]:
    """Return the correct quality arguments for the given codec."""
    if is_nvenc_codec(codec):
        return ["-cq", str(quality), "-preset", "p4"]
    return ["-crf", str(quality)]


def get_software_fallback(codec: str) -> str:
    """Return the CPU software fallback for an NVENC codec."""
    if codec == "h264_nvenc":
        return "libx264"
    if codec == "hevc_nvenc":
        return "libx265"
    return codec


def resolve_codec(codec: str) -> str:
    """Resolve 'auto' codec or validate NVENC availability, falling back to software."""
    if codec == "auto":
        if detect_nvenc():
            return "h264_nvenc"
        return "libx264"
    if is_nvenc_codec(codec) and not detect_nvenc():
        fallback = get_software_fallback(codec)
        print(f"NVENC not available, falling back from {codec} to {fallback}")
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
    except Exception as e:
        print("Running ffmpeg failed! Commandline:")
        print(" ".join(commands))
    return False


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
            "-vsync",
            "0",
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
    if not run_ffmpeg(args) and is_nvenc_codec(codec):
        fallback = get_software_fallback(codec)
        print(f"NVENC encoding failed, retrying with {fallback}")
        args[args.index(codec)] = fallback
        # Replace NVENC quality args with software quality args
        if "-cq" in args:
            cq_idx = args.index("-cq")
            quality_val = args[cq_idx + 1]
            # Remove -cq, value, -preset, preset_value
            del args[cq_idx:cq_idx + 4]
            args[cq_idx:cq_idx] = ["-crf", quality_val]
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
    fps = util.detect_fps(gif_path)
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
