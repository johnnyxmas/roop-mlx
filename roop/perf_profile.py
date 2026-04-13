"""Single source of truth for performance-profile driven behavior.

Every hot-path call site (ONNX session creation, IO binding, face detection
cadence, provider selection) consults this module instead of hard-coding flags.
That way a user with a 6 GB card can stay on `memory_saver` while a user with
a 24 GB card flips a single dropdown to `max_speed` and gets TensorRT + IO
preallocation + tracker-assisted detection, without any call sites changing.
"""

import ctypes
import os
import platform

import onnxruntime

import roop.globals

_trt_runtime_probe_result = None

MEMORY_SAVER = "memory_saver"
BALANCED = "balanced"
MAX_SPEED = "max_speed"

VALID_PROFILES = (MEMORY_SAVER, BALANCED, MAX_SPEED)


def current_profile() -> str:
    cfg = getattr(roop.globals, "CFG", None)
    if cfg is None:
        return MEMORY_SAVER
    profile = getattr(cfg, "performance_profile", MEMORY_SAVER)
    if profile not in VALID_PROFILES:
        return MEMORY_SAVER
    return profile


def get_session_options() -> onnxruntime.SessionOptions:
    """Build ORT SessionOptions tuned for the active profile.

    memory_saver disables the arena/mem-pattern/mem-reuse allocators (lower
    peak VRAM, ~10-20% slower). balanced and max_speed leave those on.
    """
    profile = current_profile()
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    if profile == MEMORY_SAVER:
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False
    else:
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True

    return sess_options


def get_enhancer_session_options() -> onnxruntime.SessionOptions:
    """Session options for face/frame enhancement models.

    Enhancement models (GFPGAN, GPEN, CodeFormer, RestoreFormer++) run once
    per face — not thousands of times like the swap model.  Using the BFC
    arena allocators on balanced/max_speed fragments VRAM and causes OOM
    when the swap model's arena is already holding most of the budget.
    Always disable the arena for these models regardless of active profile.
    CUDA graphs are also skipped: enhancers have variable-size inputs and
    running alongside large swap sessions makes graph capture unreliable.
    """
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.enable_mem_reuse = False
    return opts


def should_preallocate_io() -> bool:
    """Preallocate CUDA input OrtValues and reuse via update_inplace."""
    return current_profile() in (BALANCED, MAX_SPEED)


def should_use_cuda_graphs() -> bool:
    """Enable CUDA graph capture for fixed-input-size ONNX sessions.

    CUDA graphs record the entire kernel launch sequence on the first run
    and replay it with near-zero CPU overhead on subsequent runs.  This
    requires fixed input shapes (which swap models satisfy) and IO binding
    (already used in the preallocated path).  ~20-40% speedup on swap models.
    """
    return current_profile() in (BALANCED, MAX_SPEED)


def get_session_options_with_cuda_graphs() -> onnxruntime.SessionOptions:
    """Session options with CUDA graph capture enabled.

    Only use this for fixed-input-shape models (swap models, enhancers)
    where the tensor dimensions never change between calls.
    """
    opts = get_session_options()
    if should_use_cuda_graphs():
        try:
            opts.add_session_config_entry("enable_cuda_graph", "1")
        except Exception:
            pass  # older ORT versions may not support this
    return opts


def should_use_fp16_models() -> bool:
    """Use FP16-converted ONNX models when available."""
    return current_profile() in (BALANCED, MAX_SPEED)


def detector_stride() -> int:
    """How often face detection runs in video mode.

    1 = detect every frame (safest). Higher values reuse the previous frame's
    face bboxes between detections and run a cheap scene-cut check to force
    redetection when the frame changes drastically.
    """
    profile = current_profile()
    if profile == MEMORY_SAVER:
        return 1
    cfg = getattr(roop.globals, "CFG", None)
    configured = getattr(cfg, "detector_stride", 3) if cfg else 3
    if profile == BALANCED:
        return max(1, min(2, int(configured)))
    return max(1, int(configured))


def use_yolo_detector() -> bool:
    """Use YOLOv8n-Face on memory_saver; InsightFace on balanced/max_speed."""
    return current_profile() == MEMORY_SAVER


def scene_cut_threshold() -> float:
    """Mean absolute diff (0..255 domain) above which we force redetection."""
    return 8.0


def trt_cache_dir() -> str:
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "trt_cache")
    )
    os.makedirs(path, exist_ok=True)
    return path


def _probe_trt_runtime() -> bool:
    """Return True only if the real TensorRT runtime is loadable.

    ``onnxruntime.get_available_providers()`` reports TensorrtExecutionProvider
    whenever ORT's thin wrapper DLL exists, which is a false positive when the
    actual TensorRT runtime (``nvinfer_10.dll`` on Windows, ``libnvinfer.so``
    on Linux) isn't on the library search path. Asking ORT to use TRT in that
    state produces the slow provider_bridge retry and can leave session state
    in a broken condition. We probe ctypes once and cache the result.
    """
    global _trt_runtime_probe_result
    if _trt_runtime_probe_result is not None:
        return _trt_runtime_probe_result

    try:
        available = onnxruntime.get_available_providers()
    except Exception:
        _trt_runtime_probe_result = False
        return False
    if "TensorrtExecutionProvider" not in available:
        _trt_runtime_probe_result = False
        return False

    if platform.system().lower() == "windows":
        candidates = ("nvinfer_10.dll", "nvinfer.dll")
        loader = ctypes.WinDLL
    else:
        candidates = ("libnvinfer.so.10", "libnvinfer.so")
        loader = ctypes.CDLL

    for name in candidates:
        try:
            loader(name)
            _trt_runtime_probe_result = True
            return True
        except OSError:
            continue

    print(
        "[perf_profile] TensorRT requested by max_speed profile but the "
        "TensorRT runtime DLL could not be loaded — falling back to CUDA. "
        "Install TensorRT and add its lib directory to PATH to enable "
        "TensorRT acceleration."
    )
    _trt_runtime_probe_result = False
    return False


def augment_providers(providers):
    """Prepend TensorrtExecutionProvider when profile is max_speed and TRT is
    genuinely loadable. Fail-soft in every other case so the app still runs.
    """
    if current_profile() != MAX_SPEED:
        return providers
    if not _probe_trt_runtime():
        return providers

    device_id = int(getattr(roop.globals, "cuda_device_id", 0) or 0)
    trt_entry = (
        "TensorrtExecutionProvider",
        {
            "device_id": device_id,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache_dir(),
            "trt_timing_cache_enable": True,
        },
    )

    already_trt = any(
        (isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider")
        or p == "TensorrtExecutionProvider"
        for p in providers
    )
    if already_trt:
        return providers
    return [trt_entry] + list(providers)


def trt_available_for_max_speed() -> bool:
    if current_profile() != MAX_SPEED:
        return False
    return _probe_trt_runtime()
