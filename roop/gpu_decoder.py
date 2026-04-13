"""Hardware-accelerated video decoding via FFmpeg subprocess.

Replaces ``cv2.VideoCapture`` in the in-memory pipeline with an FFmpeg
process that uses NVDEC / DXVA2 / VAAPI (whichever the system has) for
decoding, then pipes raw BGR24 frames to stdout.  Falls back gracefully
to software decode when no hardware decoder is available.

The API mirrors the subset of ``cv2.VideoCapture`` that ``ProcessMgr``
actually uses so it can be a drop-in replacement.
"""

import shutil
import subprocess
import numpy as np


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    return path


def _probe_video(path: str):
    """Return (width, height, fps, frame_count) via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "csv=p=0",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10)
        parts = out.decode().strip().split(",")
        w = int(parts[0])
        h = int(parts[1])
        # r_frame_rate is "num/den"
        rfr = parts[2]
        if "/" in rfr:
            num, den = rfr.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(rfr)
        nb = int(parts[3]) if parts[3] != "N/A" else 0
        return w, h, fps, nb
    except Exception:
        return None


class GPUVideoCapture:
    """FFmpeg-based video reader with hardware-accelerated decoding.

    Usage mirrors ``cv2.VideoCapture``::

        cap = GPUVideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ...
        cap.release()
    """

    def __init__(self, path: str, hwaccel: str = "auto"):
        self._path = path
        self._hwaccel = hwaccel
        self._proc = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frame_count = 0
        self._frame_size = 0
        self._start_frame = 0
        self._started = False

        # Probe video metadata
        info = _probe_video(path)
        if info:
            self._width, self._height, self._fps, self._frame_count = info
            self._frame_size = self._width * self._height * 3
        else:
            # Fall back to OpenCV for metadata
            import cv2
            cap = cv2.VideoCapture(path)
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = cap.get(cv2.CAP_PROP_FPS)
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._frame_size = self._width * self._height * 3
            cap.release()

    def get(self, prop_id):
        import cv2
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        return 0.0

    def set(self, prop_id, value):
        import cv2
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._start_frame = int(value)
        elif prop_id == cv2.CAP_PROP_BUFFERSIZE:
            pass  # not applicable for pipe-based reading

    def _start_process(self):
        if self._started:
            return

        ffmpeg = _find_ffmpeg()

        # Build input args with hardware acceleration
        input_args = []

        # Try hardware decode -- ffmpeg silently falls back to software
        # if the requested hwaccel is unavailable.
        if self._hwaccel and self._hwaccel != "none":
            input_args += ["-hwaccel", self._hwaccel]

        input_args += ["-i", self._path]

        # Seek to start frame via output filter (frame-accurate)
        if self._start_frame > 0:
            # Use -ss with time-based seeking for speed, then trim
            start_time = self._start_frame / self._fps if self._fps > 0 else 0
            # Place -ss before -i for fast seeking
            input_args = (
                ["-ss", f"{start_time:.6f}"]
                + input_args
            )

        cmd = [
            ffmpeg,
            *input_args,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-v", "error",
            "pipe:1",
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_size * 2,
        )
        self._started = True

    def read(self):
        """Read the next frame.  Returns ``(True, frame)`` or ``(False, None)``."""
        self._start_process()

        raw = self._proc.stdout.read(self._frame_size)
        if len(raw) < self._frame_size:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )
        # Return a writable copy (the buffer from Popen may be read-only)
        return True, frame.copy()

    def release(self):
        if self._proc is not None:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._started = False

    def isOpened(self):
        return self._frame_size > 0


def gpu_decode_available() -> bool:
    """Quick check: is ffmpeg available on PATH?"""
    return shutil.which("ffmpeg") is not None
