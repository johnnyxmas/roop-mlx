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
    except:
        print(f"Exception reading {filename}")
    return None


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    global current_video_path, current_capture, current_frame_total

    if video_path != current_video_path:
        release_video()
        current_capture = cv2.VideoCapture(video_path)
        
        # Optimize video capture settings for better performance
        current_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for faster seeking
        
        current_video_path = video_path
        current_frame_total = current_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    current_capture.set(
        cv2.CAP_PROP_POS_FRAMES, min(current_frame_total, frame_number - 1)
    )
    has_frame, frame = current_capture.read()
    if has_frame:
        return frame
    return None


def release_video():
    global current_capture

    if current_capture is not None:
        current_capture.release()
        current_capture = None


def get_video_frame_total(video_path: str) -> int:
    # Optimize video capture for faster frame counting
    capture = cv2.VideoCapture(video_path)
    
    # Set buffer size to 1 for faster seeking
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
