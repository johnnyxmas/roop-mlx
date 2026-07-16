"""Detector-cadence knobs for the MLX fork.

Controls detection frequency and scene-cut sensitivity; the
heavy ORT/TRT machinery from the upstream repo is gone.
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


