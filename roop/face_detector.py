"""Face detection -- MLX backend.

YOLOv8n-Face runs through the OnnxMLXSession graph executor.  The
decode logic (letterbox, NMS, landmark mapping) is pure NumPy/OpenCV.
Recognition (ArcFace w600k_r50) and 106-point landmarks (2d106det)
are loaded lazily and only when the pipeline needs them.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from roop.utilities import resolve_relative_path


# ---------------------------------------------------------------------------
# FaceCompat
# ---------------------------------------------------------------------------
class FaceCompat:
    """Drop-in replacement for ``insightface.app.common.Face``.

    Supports both attribute access (``face.bbox``) and dict-style access
    (``face["bbox"]``) used interchangeably throughout the codebase.
    """

    __slots__ = (
        "bbox", "kps", "det_score",
        "landmark_2d_106", "landmark_3d_68",
        "embedding", "normed_embedding",
        "sex", "age",
        "matrix", "mask_offsets",
        "rotation",
    )

    def __init__(self):
        # In-plane rotation (degrees, for face_util.transform) that makes this
        # face upright: 0 normally, +/-90 when it was found on a rotated frame.
        self.rotation = 0
        self.bbox = None
        self.kps = None
        self.det_score = 0.0
        self.landmark_2d_106 = None
        self.landmark_3d_68 = None
        self.embedding = None
        self.normed_embedding = None
        self.sex = None
        self.age = None
        self.matrix = None
        self.mask_offsets = (0, 0, 0, 0, 1, 20)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


# ---------------------------------------------------------------------------
# YOLOFaceDetector
# ---------------------------------------------------------------------------
class YOLOFaceDetector:
    """YOLOv8n-Face detector backed by the MLX ONNX executor."""

    MODEL_FILE = "yoloface_8n.onnx"
    INPUT_SIZE = (640, 640)
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    # YOLOv8n-Face is trained on upright faces.  Measured 2026-11-05 on a
    # 1080p frame: a 322px face is lost entirely beyond ~45-60 deg of roll,
    # and a large face found at 90 deg has landmarks off by ~0.3 inter-eye
    # distances with 10x the frame-to-frame jitter -- the "jerky horizontal
    # face" symptom.  Past this roll (or when nothing is found) detection is
    # retried on the frame rotated 90 deg each way; cv2.rotate is a lossless
    # pixel permutation, so the only cost is the extra detector passes
    # (~3 ms each), and only on frames that need them.
    ROLL_RETRY_DEG = 45.0
    # Clean upright faces score ~0.85; below this the upright pass may have
    # misread an off-axis face, so the rotated passes run as well.
    RETRY_BELOW_SCORE = 0.7
    # (cv2 rotate code, face_util.transform rotation that uprights the face)
    _RETRY_ROTATIONS = (
        (cv2.ROTATE_90_CLOCKWISE, 90),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, -90),
    )

    def __init__(self):
        self._session = None
        self._recognition_session = None
        self._landmark_2d106_session = None
        self._genderage_session = None
        self._input_name = None
        self._output_names = None

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self):
        if self._session is not None:
            return
        from roop.inference import create_session

        model_path = resolve_relative_path(f"../models/{self.MODEL_FILE}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO face model not found at {model_path}. "
                "It should be downloaded during startup."
            )
        self._session = create_session(model_path)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def _ensure_recognition(self):
        if self._recognition_session is not None:
            return
        from roop.inference import create_session

        model_path = resolve_relative_path("../models/buffalo_l/w600k_r50.onnx")
        if not os.path.exists(model_path):
            return
        self._recognition_session = create_session(model_path)

    def _ensure_landmark_2d106(self):
        if self._landmark_2d106_session is not None:
            return
        from roop.inference import create_session

        model_path = resolve_relative_path("../models/buffalo_l/2d106det.onnx")
        if not os.path.exists(model_path):
            return
        self._landmark_2d106_session = create_session(model_path)

    def _ensure_genderage(self):
        if self._genderage_session is not None:
            return
        from roop.inference import create_session

        model_path = resolve_relative_path("../models/buffalo_l/genderage.onnx")
        if not os.path.exists(model_path):
            return
        self._genderage_session = create_session(model_path)

    # -- preprocessing / postprocessing ------------------------------------

    def _preprocess(self, frame: np.ndarray):
        """Letterbox-resize *frame* to INPUT_SIZE and return the blob plus
        inverse-mapping parameters."""
        h, w = frame.shape[:2]
        target_h, target_w = self.INPUT_SIZE

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        blob = np.ascontiguousarray(
            padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        )
        return blob, scale, pad_w, pad_h

    def _postprocess(self, outputs, scale, pad_w, pad_h, orig_h, orig_w):
        """Decode YOLO output tensor into a list of FaceCompat objects."""
        output = outputs[0]

        # YOLOv8 raw shape: [1, C, N]; transpose to [1, N, C]
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)
        output = output[0]  # drop batch dim → [N, C]

        num_ch = output.shape[1]
        boxes_cxcywh = output[:, :4]
        scores = output[:, 4]
        # YOLOv8-Face: 20 ch = 4 bbox + 1 conf + 5×(x,y,vis); older 15-ch = 5×(x,y)
        has_landmarks = num_ch >= 15
        kps_per_point = 3 if num_ch >= 20 else 2
        if has_landmarks:
            kps_cols = 5 * kps_per_point
            landmarks_raw = output[:, 5:5 + kps_cols].reshape(-1, 5, kps_per_point)[:, :, :2]

        # confidence filter
        mask = scores > self.CONF_THRESHOLD
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        if has_landmarks:
            landmarks_raw = landmarks_raw[mask]
        if len(boxes_cxcywh) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2
        half_w = boxes_cxcywh[:, 2] / 2
        half_h = boxes_cxcywh[:, 3] / 2
        bboxes = np.column_stack([
            boxes_cxcywh[:, 0] - half_w,
            boxes_cxcywh[:, 1] - half_h,
            boxes_cxcywh[:, 0] + half_w,
            boxes_cxcywh[:, 1] + half_h,
        ])

        # NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), scores.tolist(),
            self.CONF_THRESHOLD, self.IOU_THRESHOLD,
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        faces = []
        for i in indices:
            f = FaceCompat()

            bb = bboxes[i].copy()
            bb[0] = np.clip((bb[0] - pad_w) / scale, 0, orig_w)
            bb[1] = np.clip((bb[1] - pad_h) / scale, 0, orig_h)
            bb[2] = np.clip((bb[2] - pad_w) / scale, 0, orig_w)
            bb[3] = np.clip((bb[3] - pad_h) / scale, 0, orig_h)
            f.bbox = bb.astype(np.float32)

            if has_landmarks:
                kps = landmarks_raw[i].copy()
                kps[:, 0] = (kps[:, 0] - pad_w) / scale
                kps[:, 1] = (kps[:, 1] - pad_h) / scale
                f.kps = kps.astype(np.float32)

            f.det_score = float(scores[i])
            faces.append(f)
        return faces

    # -- public API ---------------------------------------------------------

    def detect(self, frame: np.ndarray, need_recognition: bool = False,
               need_landmarks_106: bool = False, need_genderage: bool = False):
        """Detect faces in *frame* (BGR image).

        Returns
        -------
        list[FaceCompat] sorted left-to-right, or None when no faces found.
        """
        self._ensure_loaded()
        # (face, residual roll as seen by the detector in its own orientation)
        found = self._detect_oriented(frame, None, 0)
        if self._needs_retry(found):
            for code, rotation in self._RETRY_ROTATIONS:
                self._merge(found, self._detect_oriented(frame, code, rotation))
            # Still nothing seen near upright: the face may be upside down.
            if not found or any(r > self.ROLL_RETRY_DEG for _, r in found):
                self._merge(found, self._detect_oriented(frame, cv2.ROTATE_180, 180))
        faces = [f for f, _ in found]
        if not faces:
            return None

        if need_recognition:
            self._compute_embeddings(frame, faces)
        if need_landmarks_106:
            self._compute_landmarks_106(frame, faces)
        if need_genderage:
            self._compute_genderage(frame, faces)

        faces.sort(key=lambda f: f.bbox[0])
        return faces

    def _detect_oriented(self, frame: np.ndarray, code, rotation: int) -> list:
        """One detector pass on *frame* rotated by cv2 *code* (None = as is).

        Returns ``(face, roll)`` pairs with bbox/kps mapped back to *frame*
        coordinates; ``roll`` is measured in the rotated image, i.e. how far
        from upright the detector actually saw the face.
        """
        img = frame if code is None else cv2.rotate(frame, code)
        ih, iw = img.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(img)
        outs = self._session.run(self._output_names, {self._input_name: blob})
        faces = self._postprocess(outs, scale, pad_w, pad_h, ih, iw)
        rolls = [self._roll(f) for f in faces]
        if code is None:
            return list(zip(faces, rolls))

        h, w = frame.shape[:2]

        def back(pts):
            u, v = pts[:, 0], pts[:, 1]
            if code == cv2.ROTATE_90_CLOCKWISE:      # (x, y) -> (h-1-y, x)
                return np.stack([v, h - 1 - u], axis=1)
            if code == cv2.ROTATE_180:               # (x, y) -> (w-1-x, h-1-y)
                return np.stack([w - 1 - u, h - 1 - v], axis=1)
            return np.stack([w - 1 - v, u], axis=1)  # (x, y) -> (y, w-1-x)

        for f in faces:
            x1, y1, x2, y2 = f.bbox
            corners = back(np.array([[x1, y1], [x2, y2]], dtype=np.float32))
            f.bbox = np.concatenate(
                [corners.min(axis=0), corners.max(axis=0)]
            ).astype(np.float32)
            if f.kps is not None:
                f.kps = back(f.kps).astype(np.float32)
            f.rotation = rotation
        return list(zip(faces, rolls))

    def _needs_retry(self, found: list) -> bool:
        """Should the rotated passes run?  When nothing was found, a face looks
        rolled past ROLL_RETRY_DEG, or confidence is low -- far off-axis the
        upright pass can misread the roll itself (a face at 120 deg read as
        41 deg, score 0.62), and the low score is then the only tell."""
        return (not found
                or any(r > self.ROLL_RETRY_DEG for _, r in found)
                or any(f.det_score < self.RETRY_BELOW_SCORE for f, _ in found))

    def _merge(self, found: list, candidates: list) -> None:
        """Fold rotated-pass candidates into *found* (in place), keeping the
        better reading of each face and adding faces seen only rotated."""
        for g, g_roll in candidates:
            j = next((i for i, (f, _) in enumerate(found)
                      if self._iou(f.bbox, g.bbox) > 0.3), None)
            if j is None:
                found.append((g, g_roll))
            elif self._better(g, g_roll, *found[j]):
                found[j] = (g, g_roll)

    @staticmethod
    def _better(a, a_roll: float, b, b_roll: float) -> bool:
        """Is candidate *a* a better reading of the same face than *b*?

        The detector is most accurate on upright faces, so the orientation
        that saw the face closest to upright wins.  Confidence alone is a poor
        guide off-axis: at 120 deg of roll the wrong orientation can score
        higher yet land landmarks 0.25 inter-eye distances off.
        """
        if abs(a_roll - b_roll) > 15.0:
            return a_roll < b_roll
        return a.det_score > b.det_score

    @staticmethod
    def _roll(face) -> float:
        """Absolute in-plane roll in degrees from the eye line (0 = upright)."""
        if face.kps is None:
            return 0.0
        dx, dy = face.kps[1] - face.kps[0]
        return abs(float(np.degrees(np.arctan2(dy, dx))))

    @staticmethod
    def _iou(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    # -- sub-model helpers -------------------------------------------------

    def _compute_embeddings(self, frame: np.ndarray, faces: list):
        """Run ArcFace w600k_r50 on aligned crops to fill .embedding."""
        self._ensure_recognition()
        if self._recognition_session is None:
            return
        from roop.face_util import estimate_norm

        inp_name = self._recognition_session.get_inputs()[0].name
        out_name = self._recognition_session.get_outputs()[0].name

        for face in faces:
            if face.kps is None:
                continue
            M = estimate_norm(face.kps, 112)
            aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
            aligned = (aligned / 127.5) - 1.0
            blob = np.ascontiguousarray(
                aligned.transpose(2, 0, 1)[np.newaxis]
            )
            emb = self._recognition_session.run(
                [out_name], {inp_name: blob}
            )[0][0]
            face.embedding = emb
            face.normed_embedding = emb / (np.linalg.norm(emb) + 1e-6)

    def _compute_landmarks_106(self, frame: np.ndarray, faces: list):
        """Run InsightFace 2d106det model on each detected face."""
        self._ensure_landmark_2d106()
        if self._landmark_2d106_session is None:
            return
        from roop.face_util import transform as face_transform, trans_points

        inp_name = self._landmark_2d106_session.get_inputs()[0].name
        out_name = self._landmark_2d106_session.get_outputs()[0].name
        input_size = 192

        for face in faces:
            if face.bbox is None:
                continue
            x1, y1, x2, y2 = face.bbox
            w = x2 - x1
            h = y2 - y1
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            _scale = input_size / (max(w, h) * 1.5)

            aimg, M = face_transform(frame, center, input_size, _scale,
                                     getattr(face, "rotation", 0))
            blob = cv2.dnn.blobFromImage(
                aimg, 1.0, (input_size, input_size),
                (0.0, 0.0, 0.0), swapRB=True,
            )
            pred = self._landmark_2d106_session.run(
                [out_name], {inp_name: blob}
            )[0][0]
            # 2d106det outputs [-1, 1] crop coordinates; scale to the 192px
            # crop before mapping back (insightface's Landmark.get does the
            # same).  Without this every point collapsed into a ~3px blob near
            # the crop's top-left corner -- off the face, often off-frame --
            # which broke autorotate's direction choice and mouth restore.
            pred = (pred.reshape(-1, 2) + 1) * (input_size // 2)
            IM = cv2.invertAffineTransform(M)
            pred = trans_points(pred, IM)
            face.landmark_2d_106 = pred.astype(np.float32)

    def _compute_genderage(self, frame: np.ndarray, faces: list):
        """Run InsightFace genderage on each face to fill .sex and .age."""
        self._ensure_genderage()
        if self._genderage_session is None:
            return
        from roop.face_util import transform as face_transform

        inp_name = self._genderage_session.get_inputs()[0].name
        out_name = self._genderage_session.get_outputs()[0].name
        input_size = 96

        for face in faces:
            if face.bbox is None:
                continue
            x1, y1, x2, y2 = face.bbox
            w = x2 - x1
            h = y2 - y1
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            _scale = input_size / (max(w, h) * 1.5)

            aimg, _M = face_transform(frame, center, input_size, _scale,
                                      getattr(face, "rotation", 0))
            blob = cv2.dnn.blobFromImage(
                aimg, 1.0, (input_size, input_size),
                (0.0, 0.0, 0.0), swapRB=True,
            )
            pred = self._genderage_session.run(
                [out_name], {inp_name: blob}
            )[0][0]
            # pred: [female_logit, male_logit, age/100]
            face.sex = "M" if int(np.argmax(pred[:2])) == 1 else "F"
            face.age = int(np.round(pred[2] * 100))

    # -- cleanup -----------------------------------------------------------

    def release(self):
        self._session = None
        self._recognition_session = None
        self._landmark_2d106_session = None
        self._genderage_session = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_yolo_detector: YOLOFaceDetector | None = None


def get_yolo_detector() -> YOLOFaceDetector:
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YOLOFaceDetector()
    return _yolo_detector


def release_yolo_detector():
    global _yolo_detector
    if _yolo_detector is not None:
        _yolo_detector.release()
        _yolo_detector = None
