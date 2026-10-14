"""
SignBridge AI - Dataset Inspection Tool
Audits the ASL dataset: validates folder layout, file integrity, sample dimensions,
class distributions, and potential imbalance.
"""

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import auto_detect_dataset_path, auto_detect_test_path
from src.data_loader import discover_dataset_structure, resolve_training_directory


def inspect_asl_dataset() -> None:
    print("=" * 70)
    print("🔎 SignBridge AI - Kaggle ASL Alphabet Dataset Inspector")
    print("=" * 70)

    try:
        resolved_train = resolve_training_directory()
        print(f"✅ Found Training Directory: {resolved_train}")
    except Exception as e:
        print(f"❌ Could not automatically locate training directory: {e}")
        return

    test_path = auto_detect_test_path()
    if test_path.exists():
        print(f"✅ Found Test Directory:     {test_path}")
    else:
        print(f"⚠️ Test directory not found at standard paths. (Evaluation split will be used).")

    info = discover_dataset_structure(resolved_train)
    classes = info["classes"]
    counts = info["counts"]

    print("\n📊 Dataset Overview:")
    print(f"• Total Valid Classes: {info['num_classes']}")
    print(f"• Total Image Files:   {info['total_images']:,}")
    print(f"• Balanced Structure:  {'YES' if info['is_balanced'] else 'NO'}")
    print(f"• Min per class:       {info['min_samples_per_class']:,}")
    print(f"• Max per class:       {info['max_samples_per_class']:,}")

    # Inspect image dimensions and check integrity across classes
    print("\n🔍 Sampling Images for Dimension & Channel Verification:")
    sample_shapes = {}
    corrupt_count = 0

    for cls in classes:
        cls_dir = resolved_train / cls
        sample_files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        if sample_files:
            # Check first image
            first_img = sample_files[0]
            mat = cv2.imread(str(first_img))
            if mat is None:
                corrupt_count += 1
            else:
                shape_str = f"{mat.shape[1]}x{mat.shape[0]} (Channels: {mat.shape[2]})"
                sample_shapes[shape_str] = sample_shapes.get(shape_str, 0) + 1

    print("• Detected Image Resolutions:")
    for res, num in sample_shapes.items():
        print(f"  - {res}: across {num} sample classes")

    if corrupt_count > 0:
        print(f"⚠️ Warning: Found {corrupt_count} unreadable images!")
    else:
        print("✅ Sampled images read cleanly with zero corruptions detected.")

    # Class Breakdown Table
    print("\n📋 Class Distribution Breakdown:")
    col_width = 10
    row_str = ""
    for i, cls in enumerate(classes, start=1):
        cnt = counts.get(cls, 0)
        row_str += f"{cls.upper():<6}: {cnt:>5}  "
        if i % 4 == 0:
            print("  " + row_str)
            row_str = ""
    if row_str:
        print("  " + row_str)

    # Kaggle test directory anomaly check
    if test_path.exists():
        test_files = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
        print(f"\n⚠️ Note on Kaggle Test Set:")
        print(f"• Kaggle test directory contains only {len(test_files)} sample files.")
        print(f"  In the original dataset, each letter has only 1 sample image in the test folder.")
        print(f"  SignBridge AI creates a stratified held-out validation/test split from the 87,000 images")
        print(f"  to provide statistically significant evaluation metrics across all 29 classes.")

    print("\n" + "=" * 70)
    print("✅ Dataset inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    inspect_asl_dataset()
