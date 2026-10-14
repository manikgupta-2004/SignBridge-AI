"""
Unit tests for SignBridge AI Preprocessing Module.
"""

import numpy as np
import pytest
from src.preprocessing import normalize_image, get_center_roi, extract_hand_roi


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarkList:
    def __init__(self, points):
        self.landmark = [MockLandmark(x, y) for x, y in points]


def test_normalize_image_shape_and_type():
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    norm = normalize_image(dummy_img, target_size=(224, 224), model_type="efficientnet_b0")

    assert norm.shape == (1, 224, 224, 3)
    assert norm.dtype == np.float32


def test_normalize_image_empty_raises_value_error():
    empty_img = np.array([])
    with pytest.raises(ValueError):
        normalize_image(empty_img)


def test_get_center_roi():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi, bbox = get_center_roi(frame, box_ratio=0.5)

    x_min, y_min, x_max, y_max = bbox
    assert x_max > x_min
    assert y_max > y_min
    assert roi.shape[0] == y_max - y_min
    assert roi.shape[1] == x_max - x_min
    assert roi.shape[0] == roi.shape[1]  # Must be square


def test_extract_hand_roi_with_mocked_landmarks():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Simulated hand centered around (0.5, 0.5)
    points = [
        (0.50, 0.50), (0.52, 0.45), (0.54, 0.40),
        (0.48, 0.42), (0.46, 0.48), (0.51, 0.55)
    ]
    mock_landmarks = MockLandmarkList(points)

    roi, bbox = extract_hand_roi(frame, mock_landmarks, padding_ratio=0.25)

    assert roi is not None
    x_min, y_min, x_max, y_max = bbox
    assert 0 <= x_min < x_max <= 1280
    assert 0 <= y_min < y_max <= 720
    assert roi.shape[0] > 0 and roi.shape[1] > 0
