from typing import Optional
import cv2
import numpy as np

from roop.typing import Frame

current_video_path = None
current_frame_total = 0
current_capture = None


def get_image_frame(filename: str):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Exception reading {filename}: {e}")
    return None


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    """Open *video_path* with the fastest correct backend.

    AVFoundation decodes through VideoToolbox hardware (~2.5x faster for
    4K HEVC) and honors colorspace flags, which OpenCV's default FFmpeg
    backend does not (it always converts with BT.601).  Falls back to the
    default backend for anything AVFoundation can't read (GIFs, exotic
    containers).
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ok, frame = cap.read()
        if (
            ok
            and frame is not None
            and frame.size > 0
            and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        cap.release()
    return cv2.VideoCapture(video_path)


def _open_capture(video_path: str):
    """Open a VideoCapture and cache it."""
    global current_capture, current_video_path, current_frame_total
    release_video()
    cap = open_video_capture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    current_capture = cap
    current_video_path = video_path
    current_frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    if video_path != current_video_path:
        _open_capture(video_path)

    target = min(int(current_frame_total) - 1, max(0, frame_number - 1))
    current_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    has_frame, frame = current_capture.read()
    if has_frame and frame is not None and frame.size > 0:
        return frame

    # Seek may have put the codec into a bad state (e.g. B-frame reference
    # missing after a direct seek in H.264).  Reopen and retry once.
    try:
        _open_capture(video_path)
        current_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        has_frame, frame = current_capture.read()
        if has_frame and frame is not None and frame.size > 0:
            return frame
    except Exception:
        pass
    return None


def release_video():
    global current_capture

    if current_capture is not None:
        current_capture.release()
        current_capture = None


def get_video_frame_total(video_path: str) -> int:
    # Same backend as the processing readers so frame counts stay consistent.
    capture = open_video_capture(video_path)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
