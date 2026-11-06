"""Detector pipeline plumbing: detection, embeddings, landmarks, genderage."""

import numpy as np
import pytest

from conftest import requires_model


@pytest.fixture(scope="module")
def detector():
    from roop.face_detector import get_yolo_detector

    return get_yolo_detector()


@pytest.fixture()
def noise_frame(rng):
    return rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8).copy()


@pytest.fixture()
def synthetic_face():
    from roop.face_detector import FaceCompat

    f = FaceCompat()
    f.bbox = np.array([500, 200, 700, 460], dtype=np.float32)
    f.kps = np.array(
        [[560, 300], [640, 300], [600, 350], [570, 400], [630, 400]],
        dtype=np.float32,
    )
    return f


@requires_model("yoloface_8n.onnx")
def test_detect_noise_returns_none_without_crashing(detector, noise_frame):
    result = detector.detect(
        noise_frame, need_recognition=True, need_landmarks_106=True,
        need_genderage=True,
    )
    assert result is None


@requires_model("buffalo_l", "w600k_r50.onnx")
def test_embedding_shape_and_norm(detector, noise_frame, synthetic_face):
    detector._compute_embeddings(noise_frame, [synthetic_face])
    assert synthetic_face.embedding.shape == (512,)
    assert abs(float(np.linalg.norm(synthetic_face.normed_embedding)) - 1.0) < 1e-4


@requires_model("buffalo_l", "2d106det.onnx")
def test_landmarks_106_shape(detector, noise_frame, synthetic_face):
    detector._compute_landmarks_106(noise_frame, [synthetic_face])
    assert synthetic_face.landmark_2d_106.shape == (106, 2)


@requires_model("buffalo_l", "genderage.onnx")
def test_genderage_populates_sex_and_age(detector, noise_frame, synthetic_face):
    detector._compute_genderage(noise_frame, [synthetic_face])
    assert synthetic_face.sex in ("M", "F")
    assert isinstance(synthetic_face.age, int)
    assert 0 <= synthetic_face.age <= 150


@requires_model("buffalo_l", "2d106det.onnx")
def test_landmarks_106_span_the_face(detector, noise_frame, synthetic_face):
    """Regression: the [-1, 1] model output was never scaled to the crop, so
    all 106 points collapsed into a ~3px blob off the face (often off-frame).
    Even on noise, correctly scaled points spread across the face box."""
    detector._compute_landmarks_106(noise_frame, [synthetic_face])
    lm = synthetic_face.landmark_2d_106
    x1, y1, x2, y2 = synthetic_face.bbox
    w, h = x2 - x1, y2 - y1
    assert np.ptp(lm[:, 0]) > 0.25 * w and np.ptp(lm[:, 1]) > 0.25 * h
    assert lm[:, 0].min() > x1 - w and lm[:, 0].max() < x2 + w
    assert lm[:, 1].min() > y1 - h and lm[:, 1].max() < y2 + h


@pytest.mark.parametrize("code_name,rotation", [
    ("ROTATE_90_CLOCKWISE", 90),
    ("ROTATE_90_COUNTERCLOCKWISE", -90),
    ("ROTATE_180", 180),
])
def test_rotated_pass_maps_points_back(code_name, rotation):
    """A point found on the rotated frame maps back to the pixel it came from.
    The detector is stubbed: this checks the geometry, not the model."""
    import cv2
    from roop.face_detector import FaceCompat, YOLOFaceDetector

    code = getattr(cv2, code_name)
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    src = np.array([[37.0, 21.0], [120.0, 64.0]], dtype=np.float32)
    marks = []
    for x, y in src.astype(int):            # where does each pixel land?
        one = np.zeros(frame.shape[:2], dtype=np.uint8)
        one[y, x] = 255
        ry, rx = np.unravel_index(np.argmax(cv2.rotate(one, code)), one.shape if
                                  code == cv2.ROTATE_180 else one.shape[::-1])
        marks.append([rx, ry])
    marks = np.array(marks, dtype=np.float32)

    det = YOLOFaceDetector()
    det._session = type("S", (), {"run": lambda *a, **k: None})()
    det._preprocess = lambda img: (None, 1.0, 0, 0)

    def fake_post(*_a, **_k):
        f = FaceCompat()
        f.kps = np.vstack([marks, marks[:1], marks[:1], marks[:1]])
        f.bbox = np.array([*marks.min(0), *marks.max(0)], dtype=np.float32)
        f.det_score = 0.9
        return [f]
    det._postprocess = fake_post

    (face, _roll), = det._detect_oriented(frame, code, rotation)
    np.testing.assert_allclose(face.kps[:2], src, atol=1e-4)
    np.testing.assert_allclose(face.bbox, [*src.min(0), *src.max(0)], atol=1e-4)
    assert face.rotation == rotation


def test_merge_prefers_the_more_upright_reading():
    """Far off-axis the wrong orientation can score higher; the reading that
    saw the face closest to upright must still win."""
    from roop.face_detector import FaceCompat, YOLOFaceDetector

    def face(score, rotation):
        f = FaceCompat()
        f.bbox = np.array([10, 10, 110, 130], dtype=np.float32)
        f.det_score, f.rotation = score, rotation
        return f

    det = YOLOFaceDetector()
    found = [(face(0.90, 0), 70.0)]
    det._merge(found, [(face(0.80, 90), 5.0)])
    assert found[0][0].rotation == 90
    det._merge(found, [(face(0.95, -90), 12.0)])   # within 15 deg: score decides
    assert found[0][0].rotation == -90
    det._merge(found, [(face(0.99, 180), 60.0)])   # less upright: rejected
    assert found[0][0].rotation == -90


@pytest.mark.parametrize("rotation,action", [
    (90, "rotate_clockwise"), (-90, "rotate_anticlockwise"), (0, None), (180, None),
])
def test_autorotate_follows_detector_rotation(rotation, action):
    """Autorotate turns the face the way the detector found it upright,
    rather than guessing from the box shape and 106-point landmarks."""
    from roop.ProcessMgr import ProcessMgr
    from roop.face_detector import FaceCompat

    f = FaceCompat()
    f.bbox = np.array([0, 0, 200, 100], dtype=np.float32)   # wide box
    f.rotation = rotation
    assert ProcessMgr.rotation_action(None, f, np.zeros((480, 640, 3), np.uint8)) == action
