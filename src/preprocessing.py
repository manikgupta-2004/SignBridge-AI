"""
SignBridge AI - Preprocessing Module
Provides uniform image preprocessing, normalization, and hand ROI extraction for training and inference.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np


def normalize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    model_type: str = "efficientnet_b0"
) -> np.ndarray:
    """
    Resizes and normalizes an RGB image consistently for model input.

    Args:
        image: Numpy array of shape (H, W, 3) in RGB format.
        target_size: Expected model input dimensions (width, height).
        model_type: Architecture type ('efficientnet_b0', 'mobilenet_v2', or 'standard').

    Returns:
        Batched normalized float32 tensor of shape (1, target_size[1], target_size[0], 3).
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image provided to normalize_image.")

    # Resize to target dimensions
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    resized_float = resized.astype(np.float32)

    # Apply architecture-specific normalization
    if "efficientnet" in model_type.lower():
        # EfficientNet expects [0, 255] float RGB images
        from tensorflow.keras.applications.efficientnet import preprocess_input
        processed = preprocess_input(resized_float)
    elif "mobilenet" in model_type.lower():
        # MobileNet expects [-1, 1] float RGB images
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        processed = preprocess_input(resized_float)
    else:
        # Standard [0, 1] normalization
        processed = resized_float / 255.0

    # Expand dimensions to add batch dimension (1, H, W, 3)
    if len(processed.shape) == 3:
        processed = np.expand_dims(processed, axis=0)

    return processed


def extract_hand_roi(
    frame: np.ndarray,
    hand_landmarks: Any,
    padding_ratio: float = 0.15,
    force_square: bool = True
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Extracts a padded, aspect-ratio preserved Region of Interest (ROI) centered on hand landmarks.

    Args:
        frame: BGR or RGB image of shape (H, W, 3).
        hand_landmarks: MediaPipe NormalizedLandmarkList.
        padding_ratio: Fraction of the hand size to add as padding around the bounding box.
        force_square: If True, centers a square bounding box around hand centroid.

    Returns:
        Tuple of (cropped_roi, (x_min, y_min, x_max, y_max)).
        If crop is invalid, returns (None, (0, 0, 0, 0)).
    """
    h, w = frame.shape[:2]

    # Collect pixel coordinates from 21 landmarks
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

    if not x_coords or not y_coords:
        return None, (0, 0, 0, 0)

    raw_x_min, raw_x_max = min(x_coords), max(x_coords)
    raw_y_min, raw_y_max = min(y_coords), max(y_coords)

    box_w = raw_x_max - raw_x_min
    box_h = raw_y_max - raw_y_min

    center_x = (raw_x_min + raw_x_max) / 2.0
    center_y = (raw_y_min + raw_y_max) / 2.0

    if force_square:
        side = max(box_w, box_h) * (1.0 + padding_ratio * 2)
        half_side = int(side / 2.0)
        x_min = int(center_x - half_side)
        x_max = int(center_x + half_side)
        y_min = int(center_y - half_side)
        y_max = int(center_y + half_side)
    else:
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)
        x_min = raw_x_min - pad_x
        x_max = raw_x_max + pad_x
        y_min = raw_y_min - pad_y
        y_max = raw_y_max + pad_y

    # Clamp coordinates to frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None, (0, 0, 0, 0)

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return None, (0, 0, 0, 0)

    return roi, (x_min, y_min, x_max, y_max)


def get_center_roi(
    frame: np.ndarray,
    box_ratio: float = 0.55
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extracts a centered square bounding box ROI as a fallback when no hand is detected.

    Args:
        frame: Image of shape (H, W, 3).
        box_ratio: Ratio of the frame height to use as the square box dimension.

    Returns:
        Tuple of (cropped_roi, (x_min, y_min, x_max, y_max)).
    """
    h, w = frame.shape[:2]
    side = int(min(h, w) * box_ratio)

    x_min = max(0, (w - side) // 2)
    y_min = max(0, (h - side) // 2)
    x_max = min(w, x_min + side)
    y_max = min(h, y_min + side)

    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)


def save_preprocessing_config(file_path: Path, config: Dict[str, Any]) -> None:
    """Saves preprocessing parameters to a JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def load_preprocessing_config(file_path: Path) -> Dict[str, Any]:
    """Loads preprocessing parameters from a JSON file."""
    file_path = Path(file_path)
    if not file_path.exists():
        return {
            "image_size": [224, 224],
            "model_type": "efficientnet_b0",
            "channels": 3,
            "color_mode": "RGB"
        }
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
