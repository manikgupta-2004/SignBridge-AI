"""
SignBridge AI - Inference Engine
High-performance, thread-safe sign language prediction pipeline supporting single frames,
batches, and latency benchmarking.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import CONFIG, PROJECT_ROOT
from src.preprocessing import normalize_image, load_preprocessing_config


@dataclass
class PredictionResult:
    """Encapsulates the prediction output for a single hand ROI."""
    label: str
    confidence: float
    probabilities: np.ndarray
    top_k: List[Tuple[str, float]]
    latency_ms: float


class SignPredictor:
    """
    Inference manager that handles model loading, preprocessing alignment,
    and batched or single-frame predictions.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else self._find_model_path()
        self.model: Optional[tf.keras.Model] = None
        self.class_names: List[str] = CONFIG.dataset.classes.copy()
        self.idx_to_class: Dict[int, str] = {}
        self.preprocessing_config: Dict[str, Any] = {}
        self.model_type: str = CONFIG.model.architecture
        self.target_size: Tuple[int, int] = CONFIG.dataset.image_size

        self._load_configs()
        self._load_model()

    def _find_model_path(self) -> Path:
        """Locates the trained model across standard project paths."""
        candidates = [
            CONFIG.model.trained_dir / CONFIG.model.best_model_name,
            CONFIG.model.checkpoints_dir / "best_model.keras",
            PROJECT_ROOT / "models" / "trained" / "signbridge_efficientnet.keras",
            PROJECT_ROOT / "models" / "trained" / "signbridge_mobilenet.keras",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"No trained model found. Checked: {[str(c) for c in candidates]}. "
            "Please train a model using 'python train.py' first."
        )

    def _load_configs(self) -> None:
        """Loads class indices mapping and preprocessing settings."""
        # Check for class_indices.json in model directory
        class_indices_file = self.model_path.parent / CONFIG.model.class_indices_file
        if class_indices_file.exists():
            with open(class_indices_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                # Invert {class_name: index} to {index: class_name}
                self.idx_to_class = {int(v): k for k, v in mapping.items()}
                self.class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
        else:
            # Fallback to default alphabetical ordering
            self.class_names = CONFIG.dataset.classes.copy()
            self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}

        # Check preprocessing config
        prep_file = self.model_path.parent / CONFIG.model.preprocessing_config_file
        self.preprocessing_config = load_preprocessing_config(prep_file)
        self.model_type = self.preprocessing_config.get("model_type", CONFIG.model.architecture)
        target = self.preprocessing_config.get("image_size", CONFIG.dataset.image_size)
        self.target_size = (int(target[0]), int(target[1]))

    def _load_model(self) -> None:
        """Loads and compiles the Keras model with warm-up."""
        print(f"[SignPredictor] Loading model from {self.model_path}...")
        self.model = load_model(str(self.model_path))
        print("[SignPredictor] Model loaded successfully.")

        # Warm-up inference run to avoid first-frame latency spike
        dummy_input = np.zeros((1, self.target_size[1], self.target_size[0], 3), dtype=np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        print("[SignPredictor] Warm-up inference completed.")

    def predict_roi(self, roi_image: np.ndarray, is_bgr: bool = True, top_k_num: int = 3) -> PredictionResult:
        """
        Runs inference on an extracted hand ROI.

        Args:
            roi_image: Numpy array of the cropped hand region.
            is_bgr: If True, converts BGR (OpenCV) to RGB.
            top_k_num: Number of top candidate predictions to return.

        Returns:
            PredictionResult object.
        """
        if roi_image is None or roi_image.size == 0:
            raise ValueError("Input ROI is empty or None.")

        t_start = time.perf_counter()

        # Ensure RGB ordering for neural network
        rgb_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB) if is_bgr else roi_image

        # Preprocess input tensor
        preprocessed = normalize_image(
            rgb_image,
            target_size=self.target_size,
            model_type=self.model_type
        )

        # Run forward pass
        raw_probs = self.model.predict(preprocessed, verbose=0)[0]
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Compute top prediction
        top_idx = int(np.argmax(raw_probs))
        top_label = self.idx_to_class.get(top_idx, self.class_names[top_idx])
        top_confidence = float(raw_probs[top_idx])

        # Compute top-K predictions
        sorted_indices = np.argsort(raw_probs)[::-1][:top_k_num]
        top_k = [
            (self.idx_to_class.get(int(i), self.class_names[i]), float(raw_probs[i]))
            for i in sorted_indices
        ]

        return PredictionResult(
            label=top_label,
            confidence=top_confidence,
            probabilities=raw_probs,
            top_k=top_k,
            latency_ms=latency_ms
        )

    def benchmark_latency(self, num_samples: int = 50) -> Dict[str, float]:
        """
        Measures inference speed over repeated evaluations.

        Returns:
            Dictionary with mean_ms, std_ms, p95_ms, and estimated FPS.
        """
        dummy = np.random.randint(0, 255, (self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        latencies = []

        # Warm-up iterations
        for _ in range(5):
            self.predict_roi(dummy, is_bgr=False)

        for _ in range(num_samples):
            res = self.predict_roi(dummy, is_bgr=False)
            latencies.append(res.latency_ms)

        mean_ms = float(np.mean(latencies))
        std_ms = float(np.std(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {
            "mean_latency_ms": round(mean_ms, 2),
            "std_latency_ms": round(std_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "estimated_fps": round(fps, 1),
            "num_samples": num_samples
        }
