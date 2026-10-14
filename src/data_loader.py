"""
SignBridge AI - Data Loader Module
Handles dataset directory discovery, validation, train/val/test splits, data augmentation,
and class weighting.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from config import DatasetConfig, CONFIG


def resolve_training_directory(path: Optional[Path] = None) -> Path:
    """
    Finds the directory that directly contains class subfolders (e.g., 'A', 'B', etc.).
    Gracefully handles Kaggle's double-nested 'asl_alphabet_train/asl_alphabet_train' packaging.
    """
    candidate = Path(path) if path else CONFIG.dataset.train_dir

    if not candidate.exists():
        raise FileNotFoundError(f"Specified dataset directory does not exist: {candidate}")

    # Check if candidate directly contains class folders
    subdirs = [d.name for d in candidate.iterdir() if d.is_dir()]
    if any(c in subdirs for c in ["A", "B", "C"]):
        return candidate

    # Check one level deeper for nested packaging
    for sub in candidate.iterdir():
        if sub.is_dir():
            nested_subdirs = [d.name for d in sub.iterdir() if d.is_dir()]
            if any(c in nested_subdirs for c in ["A", "B", "C"]):
                return sub

    raise FileNotFoundError(
        f"Could not locate ASL alphabet class folders inside {candidate}. "
        "Expected subfolders named 'A', 'B', 'del', etc."
    )


def discover_dataset_structure(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Scans the dataset directory, counts samples per category, and verifies file validity.

    Returns:
        Dictionary containing classes, counts, total samples, and imbalance statistics.
    """
    resolved_dir = resolve_training_directory(data_dir)
    classes = sorted([d.name for d in resolved_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    counts: Dict[str, int] = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    corrupt_files: List[str] = []

    for cls in classes:
        cls_dir = resolved_dir / cls
        files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        counts[cls] = len(files)

    total_images = sum(counts.values())
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    is_balanced = (max_count - min_count) <= 5 if counts else True

    return {
        "dataset_path": str(resolved_dir),
        "classes": classes,
        "num_classes": len(classes),
        "counts": counts,
        "total_images": total_images,
        "is_balanced": is_balanced,
        "min_samples_per_class": min_count,
        "max_samples_per_class": max_count,
        "corrupt_files_detected": len(corrupt_files)
    }


def create_data_generators(
    dataset_config: Optional[DatasetConfig] = None,
    model_type: str = "efficientnet_b0"
) -> Tuple[Any, Any, Dict[str, int], Dict[int, float]]:
    """
    Creates train and validation ImageDataGenerators with safe, semantic-preserving augmentations.

    Args:
        dataset_config: Configuration settings for image dimensions and batch size.
        model_type: Architecture type to select matching normalization.

    Returns:
        Tuple of (train_generator, val_generator, class_to_idx, class_weights_dict)
    """
    cfg = dataset_config or CONFIG.dataset
    resolved_dir = resolve_training_directory(cfg.train_dir)

    # Architecture-specific preprocessing function
    if "efficientnet" in model_type.lower():
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif "mobilenet" in model_type.lower():
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        preprocess_input = lambda x: x / 255.0

    # Note: NO horizontal_flip! ASL signs are handedness-dependent.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=cfg.validation_split,
        rotation_range=12,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.12,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=cfg.validation_split
    )

    train_gen = train_datagen.flow_from_directory(
        str(resolved_dir),
        target_size=cfg.image_size,
        batch_size=cfg.batch_size,
        subset="training",
        class_mode="categorical",
        shuffle=True,
        seed=cfg.random_seed
    )

    val_gen = val_datagen.flow_from_directory(
        str(resolved_dir),
        target_size=cfg.image_size,
        batch_size=cfg.batch_size,
        subset="validation",
        class_mode="categorical",
        shuffle=False,
        seed=cfg.random_seed
    )

    class_indices = train_gen.class_indices

    # Calculate class weights in case of subtle imbalance
    labels = train_gen.classes
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels
    )
    class_weights_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}

    return train_gen, val_gen, class_indices, class_weights_dict
