from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
startup_args = None

frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
use_batch = None
source_face_index = 0
target_face_index = 0
video_encoder = None
video_quality = None
max_memory = None
execution_threads = None
headless = None
log_level = "error"
selected_enhancer = None
subsample_size = 128
face_swap_mode = None
blend_ratio = 0.5
distance_threshold = 0.65
default_det_size = True

no_face_action = 0

processing = False

progress_status = {
    "percent": 0,
    "fps": 0.0,
    "eta_seconds": 0,
    "current_frame": 0,
    "total_frames": 0,
    "start_time": 0,
    "active": False,
}

g_desired_face_analysis = None

INPUT_FACESETS = []
TARGET_FACES = []

CFG: Settings = None
