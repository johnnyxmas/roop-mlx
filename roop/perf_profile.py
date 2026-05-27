"""Performance profile defaults for the MLX fork.

The ORT/TensorRT/CUDA-graph machinery from the upstream repo is gone.
What remains is a couple of detector-cadence knobs that are still
referenced by ProcessMgr's tracker logic.  These will likely move into
the MLX runtime config once the port is further along.
"""


def detector_stride() -> int:
    """Detection cadence for video pipelines.

    Full detection runs every N frames; optical-flow tracking fills
    the gaps.  Higher values are faster but may drift on fast motion.
    """
    return 5


def scene_cut_threshold() -> float:
    """Mean absolute diff (0..255 domain) that forces redetection."""
    return 8.0


def use_yolo_detector() -> bool:
    """YOLO will be the default detector for the MLX port."""
    return True
