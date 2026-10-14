"""
SignBridge AI - Central Configuration Module
Defines dataset locations, model hyperparameters, webcam parameters, and runtime settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Default 29 classes from Kaggle ASL Alphabet dataset
DEFAULT_CLASSES: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]

# Candidate search directories for the ASL dataset
DATASET_SEARCH_PATHS = [
    PROJECT_ROOT / "dataset" / "asl_alphabet_train" / "asl_alphabet_train",
    PROJECT_ROOT / "dataset" / "asl_alphabet_train",
    PROJECT_ROOT / "dataset" / "train",
    PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train",
    PROJECT_ROOT / "data" / "asl_alphabet_train",
]

TEST_DATASET_SEARCH_PATHS = [
    PROJECT_ROOT / "dataset" / "asl_alphabet_test" / "asl_alphabet_test",
    PROJECT_ROOT / "dataset" / "asl_alphabet_test",
    PROJECT_ROOT / "dataset" / "test",
    PROJECT_ROOT / "data" / "asl_alphabet_test",
]


def auto_detect_dataset_path() -> Path:
    """Automatically finds the training dataset directory containing class subfolders."""
    for path in DATASET_SEARCH_PATHS:
        if path.exists() and path.is_dir():
            # Check if it has subfolders like 'A', 'B'
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            if any(c in subdirs for c in ["A", "B", "C"]):
                return path
    # Fallback default
    return PROJECT_ROOT / "dataset" / "asl_alphabet_train" / "asl_alphabet_train"


def auto_detect_test_path() -> Path:
    """Automatically finds the Kaggle sample test directory if present."""
    for path in TEST_DATASET_SEARCH_PATHS:
        if path.exists() and path.is_dir():
            return path
    return PROJECT_ROOT / "dataset" / "asl_alphabet_test" / "asl_alphabet_test"


@dataclass
class DatasetConfig:
    """Configuration for data loading and preprocessing."""
    train_dir: Path = field(default_factory=auto_detect_dataset_path)
    test_dir: Path = field(default_factory=auto_detect_test_path)
    image_size: Tuple[int, int] = (224, 224)
    color_channels: int = 3
    classes: List[str] = field(default_factory=lambda: DEFAULT_CLASSES.copy())
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    batch_size: int = 32


@dataclass
class ModelConfig:
    """Configuration for model training and architecture."""
    architecture: str = "efficientnet_b0"  # Options: 'efficientnet_b0', 'mobilenet_v2'
    learning_rate: float = 1e-4
    epochs: int = 10
    fine_tune_epochs: int = 5
    fine_tune_lr: float = 1e-5
    dropout_rate: float = 0.35
    dense_units: int = 512
    models_dir: Path = PROJECT_ROOT / "models"
    checkpoints_dir: Path = PROJECT_ROOT / "models" / "checkpoints"
    trained_dir: Path = PROJECT_ROOT / "models" / "trained"
    best_model_name: str = "signbridge_efficientnet.keras"
    class_indices_file: str = "class_indices.json"
    preprocessing_config_file: str = "preprocessing_config.json"


@dataclass
class WebcamConfig:
    """Configuration for real-time webcam inference and sentence stabilization."""
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps_limit: int = 30
    mirror_camera: bool = True
    
    # MediaPipe hand detection
    use_mediapipe: bool = True
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65
    hand_roi_padding_percent: float = 0.28  # 28% padding around hand landmarks
    
    # Prediction stabilization and temporal filtering
    confidence_threshold: float = 0.78  # 78% confidence required
    smoothing_window_size: int = 9       # Frames rolling window
    stability_threshold: int = 6         # Require at least 6/9 matching predictions
    
    # Debouncing and Sentence Builder
    hold_duration_frames: int = 12       # ~0.4s to 0.5s at 25-30fps before committing sign
    cooldown_frames: int = 14            # Post-commit cooldown before allowing next sign
    require_neutral_release: bool = False # If True, requires transition to 'nothing' or no hand
    
    # Class mappings
    space_class: str = "space"
    delete_class: str = "del"
    nothing_class: str = "nothing"


@dataclass
class AppConfig:
    """Global composite configuration."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    webcam: WebcamConfig = field(default_factory=WebcamConfig)
    results_dir: Path = PROJECT_ROOT / "results"


# Global default configuration instance
CONFIG = AppConfig()
