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
