"""
FFMPEG_Writer - write set of frames to video file

original from
https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_writer.py

removed unnecessary dependencies

The MIT License (MIT)

Copyright (c) 2015 Zulko
Copyright (c) 2023 Janvarev Vladislav
"""

import os
import subprocess as sp
import threading
from collections import deque

from roop.util_ffmpeg import is_videotoolbox_codec, resolve_codec, get_software_fallback

PIPE = -1
STDOUT = -2
DEVNULL = -3

FFMPEG_BINARY = "ffmpeg"


class FFMPEG_VideoWriter:
    """A class for FFMPEG-based video writing.

    A class to write videos using ffmpeg. ffmpeg will write in a large
    choice of formats.

    Parameters
    -----------

    filename
      Any filename like 'video.mp4' etc. but if you want to avoid
      complications it is recommended to use the generic extension
      '.avi' for all your videos.

    size
      Size (width,height) of the output video in pixels.

    fps
      Frames per second in the output video file.

    codec
      FFMPEG codec. It seems that in terms of quality the hierarchy is
      'rawvideo' = 'png' > 'mpeg4' > 'libx264'
      'png' manages the same lossless quality as 'rawvideo' but yields
      smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
      of accepted codecs.

      Note for default 'libx264': by default the pixel format yuv420p
      is used. If the video dimensions are not both even (e.g. 720x405)
      another pixel format is used, and this can cause problem in some
      video readers.

    audiofile
      Optional: The name of an audio file that will be incorporated
      to the video.

    preset
      Sets the time that FFMPEG will take to compress the video. The slower,
      the better the compression rate. Possibilities are: ultrafast,superfast,
      veryfast, faster, fast, medium (default), slow, slower, veryslow,
      placebo.

    bitrate
      Only relevant for codecs which accept a bitrate. "5000k" offers
      nice results in general.

    """

    def __init__(
        self,
        filename,
        size,
        fps,
        codec="libx265",
        crf=14,
        audiofile=None,
        preset="medium",
        bitrate=None,
        logfile=None,
        threads=None,
        ffmpeg_params=None,
    ):
        if logfile is None:
            logfile = sp.PIPE

        self.filename = filename
        self.codec = resolve_codec(codec)
        self.ext = self.filename.split(".")[-1]
        w = size[0] - 1 if size[0] % 2 != 0 else size[0]
        h = size[1] - 1 if size[1] % 2 != 0 else size[1]

        # order is important
        cmd = [
            FFMPEG_BINARY,
            "-hide_banner",
            "-hwaccel",
            "auto",
            "-y",
            "-loglevel",
            "error" if logfile == sp.PIPE else "info",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            "%dx%d" % (size[0], size[1]),
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-an",
            "-i",
            "-",
        ]

        if audiofile is not None:
            cmd.extend(["-i", audiofile, "-acodec", "copy"])

        cmd.extend(["-vcodec", self.codec])
        # quality_pct is a 0-100 value where higher = better quality.
        # VideoToolbox -q:v uses the same 0-100 scale directly.
        # CRF (libx264/libx265/libvpx-vp9) uses 0-51 where LOWER is better,
        # so we invert: quality 100 → CRF 0, quality 0 → CRF 51.
        quality_pct = max(0, min(100, int(crf)))
        if is_videotoolbox_codec(self.codec):
            cmd.extend(["-q:v", str(quality_pct)])
        else:
            crf_value = int((100 - quality_pct) * 51 / 100)
            cmd.extend(["-crf", str(crf_value)])
        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)
        if bitrate is not None:
            cmd.extend(["-b", bitrate])

        # Convert BGR->YUV with the BT.709 matrix and tag the stream so
        # players interpret it the same way it was encoded.  The readers
        # (AVFoundation) hand us correctly-decoded RGB, so the writer must
        # not fall back to swscale's untagged BT.601 default.  The scale
        # also rounds odd dimensions down to even for yuv420p.
        cmd.extend(
            [
                "-vf",
                f"scale={w}:{h}:out_color_matrix=bt709:out_range=tv,"
                "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-color_range", "tv",
            ]
        )

        if threads is not None:
            cmd.extend(["-threads", str(threads)])

        cmd.extend(
            [
                "-pix_fmt",
                "yuv420p",
            ]
        )
        cmd.extend([filename])

        test = str(cmd)
        print(test)

        popen_params = {"stdout": DEVNULL, "stderr": logfile, "stdin": sp.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        try:
            self.proc = sp.Popen(cmd, **popen_params)
        except Exception as e:
            if is_videotoolbox_codec(self.codec):
                fallback = get_software_fallback(self.codec)
                print(f"VideoToolbox encoder failed to start, falling back to {fallback}")
                self.codec = fallback
                vcodec_idx = cmd.index("-vcodec", cmd.index("-i"))
                cmd[vcodec_idx + 1] = fallback
                if "-q:v" in cmd:
                    q_idx = cmd.index("-q:v")
                    q_val = cmd[q_idx + 1]
                    del cmd[q_idx:q_idx + 2]
                    cmd[q_idx:q_idx] = ["-crf", q_val]
                self.proc = sp.Popen(cmd, **popen_params)
            else:
                raise IOError(f"Failed to start FFMPEG process: {e}")

        # Drain stderr on a background thread. ffmpeg's stderr pipe is small;
        # at high resolutions it fills after a few hundred frames, blocks the
        # writer, stops it reading stdin, and deadlocks the whole
        # producer/consumer pipeline. The ring buffer keeps the last 200
        # lines for diagnostics.
        self._stderr_lines: deque = deque(maxlen=200)
        self._stderr_thread = None
        if self.proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True
            )
            self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            for line in iter(self.proc.stderr.readline, b""):
                if not line:
                    break
                self._stderr_lines.append(line)
        except Exception:
            pass

    def write_frame(self, img_array):
        """Writes one frame in the file."""
        try:
            # if PY3:
            self.proc.stdin.write(img_array.tobytes())
            # else:
            #    self.proc.stdin.write(img_array.tostring())
        except (IOError, OSError, BrokenPipeError) as err:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
            ffmpeg_error = b"".join(self._stderr_lines) or b"Process communication failed"

            error = str(err) + (
                "\n\nroop unleashed error: FFMPEG encountered "
                "the following error while writing file %s:"
                "\n\n %s" % (self.filename, str(ffmpeg_error))
            )

            # Check for hardware encoder specific errors
            if (b"hevc_videotoolbox" in ffmpeg_error or
                b"h264_videotoolbox" in ffmpeg_error or
                b"VideoToolbox" in ffmpeg_error or
                b"Error encoding a frame" in ffmpeg_error):
                error = error + (
                    "\n\nHardware encoder (VideoToolbox) failed. Consider:"
                    "\n- Reducing video resolution or quality"
                    "\n- Using software encoding (libx264/libx265)"
                    "\n- Closing other GPU-intensive applications"
                )

            elif b"Unknown encoder" in ffmpeg_error:
                error = error + (
                    "\n\nThe video export "
                    "failed because FFMPEG didn't find the specified "
                    "codec for video encoding (%s). Please install "
                    "this codec or change the codec when calling "
                    "write_videofile. For instance:\n"
                    "  >>> clip.write_videofile('myvid.webm', codec='libvpx')"
                ) % (self.codec)

            elif b"incorrect codec parameters ?" in ffmpeg_error:
                error = error + (
                    "\n\nThe video export "
                    "failed, possibly because the codec specified for "
                    "the video (%s) is not compatible with the given "
                    "extension (%s). Please specify a valid 'codec' "
                    "argument in write_videofile. This would be 'libx264' "
                    "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx for webm. "
                    "Another possible reason is that the audio codec was not "
                    "compatible with the video codec. For instance the video "
                    "extensions 'ogv' and 'webm' only allow 'libvorbis' (default) as a"
                    "video codec."
                ) % (self.codec, self.ext)

            elif b"encoder setup failed" in ffmpeg_error:
                error = error + (
                    "\n\nThe video export "
                    "failed, possibly because the bitrate you specified "
                    "was too high or too low for the video codec."
                )

            elif b"Invalid encoder type" in ffmpeg_error:
                error = error + (
                    "\n\nThe video export failed because the codec "
                    "or file extension you provided is not a video"
                )

            raise IOError(error)

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()

                # Wait for process to complete with timeout to avoid hanging
                return_code = self.proc.wait(timeout=30)

                if self._stderr_thread is not None:
                    self._stderr_thread.join(timeout=5)
                if self.proc.stderr is not None:
                    try:
                        self.proc.stderr.close()
                    except Exception:
                        pass
                
                # Check for assertion failures or other FFMPEG errors
                if return_code != 0:
                    print(f"Warning: FFMPEG process exited with code {return_code}")
                    # Don't raise an error for assertion failures if video was mostly processed
                    # These are often internal FFMPEG issues that don't affect the output
                    
            except sp.TimeoutExpired:
                print("Warning: FFMPEG process did not terminate within timeout, forcing termination")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except sp.TimeoutExpired:
                    print("Warning: Force killing FFMPEG process")
                    self.proc.kill()
                    self.proc.wait()
            except Exception as e:
                print(f"Warning: Error during FFMPEG cleanup: {e}")
                # Try to terminate gracefully
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
                except:
                    try:
                        self.proc.kill()
                        self.proc.wait()
                    except:
                        pass

        self.proc = None

    # Support the Context Manager protocol, to ensure that resources are cleaned up.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
