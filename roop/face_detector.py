"""Face detection backends: InsightFace (default) and YOLOv8n-Face (fast).

The user picks a detector in Settings. The rest of the pipeline (swap,
enhance, mask) is detector-agnostic -- it only needs Face-compatible objects
with bbox, kps, and optionally landmark_2d_106 / embedding / sex attributes.
"""

import os

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.perf_profile import augment_providers, get_session_options
from roop.utilities import resolve_relative_path


# ---------------------------------------------------------------------------
# Available detector choices (shown in Settings UI)
# ---------------------------------------------------------------------------
DETECTOR_CHOICES = ["InsightFace", "YOLOv8n-Face"]


# ---------------------------------------------------------------------------
# Lightweight Face object compatible with InsightFace's Face interface
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
    )

    def __init__(self):
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

    # dict-style access ---------------------------------------------------
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


# ---------------------------------------------------------------------------
# YOLOv8n-Face detector
# ---------------------------------------------------------------------------
class YOLOFaceDetector:
    """YOLOv8n-Face detector backed by ONNX Runtime.

    Much faster than InsightFace RetinaFace for detection + 5-point landmarks.
    Recognition (ArcFace w600k_r50) and detailed landmarks (2d106det) are
    loaded lazily and only when the pipeline actually needs them.
    """

    MODEL_FILE = "yoloface_8n.onnx"
    INPUT_SIZE = (640, 640)
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4

    def __init__(self):
        self._session = None
        self._recognition_session = None
        self._landmark_2d106_session = None
        self._input_name = None
        self._output_names = None

    # -- lazy loading -----------------------------------------------------
    def _ensure_loaded(self):
        if self._session is not None:
            return

        from roop.perf_profile import should_use_fp16_models
        from roop.model_manager import get_fp16_model_path

        model_path = resolve_relative_path(f"../models/{self.MODEL_FILE}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO face model not found at {model_path}. "
                "It should be downloaded during startup."
            )

        if should_use_fp16_models():
            model_path = get_fp16_model_path(model_path)

        sess_options = get_session_options()
        if roop.globals.CFG and roop.globals.CFG.force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = augment_providers(roop.globals.execution_providers)

        self._session = onnxruntime.InferenceSession(
            model_path, sess_options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def _ensure_recognition(self):
        if self._recognition_session is not None:
            return
        from roop.perf_profile import should_use_fp16_models
        from roop.model_manager import get_fp16_model_path

        model_path = resolve_relative_path("../models/buffalo_l/w600k_r50.onnx")
        if not os.path.exists(model_path):
            return
        if should_use_fp16_models():
            model_path = get_fp16_model_path(model_path)
        sess_options = get_session_options()
        providers = (
            ["CPUExecutionProvider"]
            if roop.globals.CFG and roop.globals.CFG.force_cpu
            else augment_providers(roop.globals.execution_providers)
        )
        self._recognition_session = onnxruntime.InferenceSession(
            model_path, sess_options, providers=providers
        )

    def _ensure_landmark_2d106(self):
        if self._landmark_2d106_session is not None:
            return
        from roop.perf_profile import should_use_fp16_models
        from roop.model_manager import get_fp16_model_path

        model_path = resolve_relative_path("../models/buffalo_l/2d106det.onnx")
        if not os.path.exists(model_path):
            return
        if should_use_fp16_models():
            model_path = get_fp16_model_path(model_path)
        sess_options = get_session_options()
        providers = (
            ["CPUExecutionProvider"]
            if roop.globals.CFG and roop.globals.CFG.force_cpu
            else augment_providers(roop.globals.execution_providers)
        )
        self._landmark_2d106_session = onnxruntime.InferenceSession(
            model_path, sess_options, providers=providers
        )

    # -- preprocessing / postprocessing -----------------------------------
    def _preprocess(self, frame):
        """Letterbox-resize *frame* to ``INPUT_SIZE`` and return the blob
        together with the inverse-mapping parameters."""
        h, w = frame.shape[:2]
        target_h, target_w = self.INPUT_SIZE

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        blob = np.ascontiguousarray(
            padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        )
        return blob, scale, pad_w, pad_h

    def _postprocess(self, outputs, scale, pad_w, pad_h, orig_h, orig_w):
        """Decode YOLO output tensor into a list of `FaceCompat` objects."""
        output = outputs[0]

        # YOLOv8 raw shape is [1, C, N]; transpose to [1, N, C]
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)
        output = output[0]  # drop batch dim -> [N, C]

        num_ch = output.shape[1]
        boxes_cxcywh = output[:, :4]
        scores = output[:, 4]
        has_landmarks = num_ch >= 15
        if has_landmarks:
            landmarks_raw = output[:, 5:15].reshape(-1, 5, 2)

        # confidence filter
        mask = scores > self.CONF_THRESHOLD
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        if has_landmarks:
            landmarks_raw = landmarks_raw[mask]
        if len(boxes_cxcywh) == 0:
            return []

        # convert cx,cy,w,h -> x1,y1,x2,y2
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

    # -- public API -------------------------------------------------------
    def detect(self, frame, need_recognition=False, need_landmarks_106=False):
        """Detect faces in *frame*.

        Parameters
        ----------
        frame : ndarray
            BGR image.
        need_recognition : bool
            If *True* compute ArcFace embeddings (needed for face matching).
        need_landmarks_106 : bool
            If *True* compute 106-point landmarks (needed for mouth restore,
            auto-rotation, DMDNet).

        Returns
        -------
        list[FaceCompat] | None
            Sorted left-to-right, or *None* when no faces found.
        """
        self._ensure_loaded()
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(frame)

        outs = self._session.run(self._output_names, {self._input_name: blob})
        faces = self._postprocess(outs, scale, pad_w, pad_h, h, w)
        if not faces:
            return None

        if need_recognition:
            self._compute_embeddings(frame, faces)
        if need_landmarks_106:
            self._compute_landmarks_106(frame, faces)

        faces.sort(key=lambda f: f.bbox[0])
        return faces

    # -- sub-model helpers ------------------------------------------------
    def _compute_embeddings(self, frame, faces):
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
            emb = self._recognition_session.run([out_name], {inp_name: blob})[0][0]
            face.embedding = emb
            face.normed_embedding = emb / np.linalg.norm(emb)

    def _compute_landmarks_106(self, frame, faces):
        """Run InsightFace 2d106det model on each detected face."""
        self._ensure_landmark_2d106()
        if self._landmark_2d106_session is None:
            return
        from roop.face_util import transform as face_transform

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

            aimg, M = face_transform(frame, center, input_size, _scale, 0)
            blob = cv2.dnn.blobFromImage(
                aimg, 1.0, (input_size, input_size),
                (0.0, 0.0, 0.0), swapRB=True,
            )
            pred = self._landmark_2d106_session.run(
                [out_name], {inp_name: blob}
            )[0][0]
            pred = pred.reshape(-1, 2)
            # map back to original coordinates
            IM = cv2.invertAffineTransform(M)
            from roop.face_util import trans_points
            pred = trans_points(pred, IM)
            face.landmark_2d_106 = pred.astype(np.float32)

    # -- cleanup ----------------------------------------------------------
    def release(self):
        for attr in ("_session", "_recognition_session", "_landmark_2d106_session"):
            sess = getattr(self, attr, None)
            if sess is not None:
                del sess
                setattr(self, attr, None)


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
