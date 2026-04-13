"""Lightweight optical-flow face tracker for inter-detection frames.

When the detector stride is > 1, the previous implementation simply reused
the old face list verbatim.  This module improves on that by estimating
per-face motion via sparse Lucas-Kanade optical flow on the 5-point
landmarks, then warping the cached bounding boxes and keypoints accordingly.

This allows strides of 10-15+ without the "face lag" artifact that raw
bbox reuse causes during head movement.
"""

import cv2
import numpy as np


class FaceTracker:
    """Track faces between detection frames using optical flow."""

    # LK optical flow parameters -- tuned for face landmarks
    _lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )

    def __init__(self):
        self._prev_gray = None
        self._prev_points = None   # [N_faces, 5, 2] float32
        self._prev_faces = None    # list of face objects
        self._frames_since_detect = 0

    def reset(self):
        self._prev_gray = None
        self._prev_points = None
        self._prev_faces = None
        self._frames_since_detect = 0

    def on_detection(self, frame, faces):
        """Called after a real detection.  Caches landmarks for tracking."""
        if faces is None or len(faces) == 0:
            self.reset()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = []
        valid_faces = []
        for f in faces:
            if f.kps is not None and len(f.kps) == 5:
                points.append(f.kps.astype(np.float32))
                valid_faces.append(f)

        if not points:
            self.reset()
            return

        self._prev_gray = gray
        self._prev_points = np.array(points, dtype=np.float32)  # [N, 5, 2]
        self._prev_faces = valid_faces
        self._frames_since_detect = 0

    def track(self, frame):
        """Estimate updated face positions from optical flow.

        Returns a *new* list of face objects with warped bbox/kps, or
        ``None`` if tracking is not possible (no previous data, all points
        lost, etc.).
        """
        if self._prev_gray is None or self._prev_faces is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n_faces = len(self._prev_faces)

        # flatten [N, 5, 2] -> [N*5, 1, 2] for calcOpticalFlowPyrLK
        prev_flat = self._prev_points.reshape(-1, 1, 2)
        next_flat, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_flat, None, **self._lk_params
        )
        if next_flat is None:
            return None

        status = status.reshape(n_faces, 5)
        next_pts = next_flat.reshape(n_faces, 5, 2)

        tracked_faces = []
        new_points = []
        new_valid_faces = []

        for idx in range(n_faces):
            good = status[idx].astype(bool)
            if good.sum() < 3:
                # Too few points tracked -- drop this face
                continue

            old_kps = self._prev_points[idx]
            new_kps = next_pts[idx]

            # Compute rigid shift from good points only
            dx = np.median(new_kps[good, 0] - old_kps[good, 0])
            dy = np.median(new_kps[good, 1] - old_kps[good, 1])

            prev_face = self._prev_faces[idx]

            # Clone face with shifted bbox and kps
            from roop.face_detector import FaceCompat
            f = FaceCompat()
            f.det_score = prev_face.det_score

            # Shift bbox
            old_bbox = prev_face.bbox.copy()
            f.bbox = np.array([
                old_bbox[0] + dx,
                old_bbox[1] + dy,
                old_bbox[2] + dx,
                old_bbox[3] + dy,
            ], dtype=np.float32)

            # Use tracked keypoints directly (more accurate than shift)
            f.kps = new_kps.astype(np.float32)

            # Copy over attributes that don't change between frames
            f.embedding = getattr(prev_face, "embedding", None)
            f.normed_embedding = getattr(prev_face, "normed_embedding", None)
            f.sex = getattr(prev_face, "sex", None)
            f.age = getattr(prev_face, "age", None)

            # Shift 106 landmarks if present
            lm106 = getattr(prev_face, "landmark_2d_106", None)
            if lm106 is not None:
                f.landmark_2d_106 = lm106 + np.array([dx, dy], dtype=np.float32)

            tracked_faces.append(f)
            new_points.append(new_kps.astype(np.float32))
            new_valid_faces.append(f)

        if not tracked_faces:
            return None

        # Update state for next track() call
        self._prev_gray = gray
        self._prev_points = np.array(new_points, dtype=np.float32)
        self._prev_faces = new_valid_faces
        self._frames_since_detect += 1

        return tracked_faces
