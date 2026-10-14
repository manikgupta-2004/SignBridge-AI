"""
SignBridge AI - 3D Landmark Feature Extraction Pipeline
Extracts invariant 3D skeletal landmark features from ASL alphabet training images
and caches them into a compressed numpy (.npz) dataset for lightning-fast training.
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG


def extract_features_from_landmarks(landmarks, scale: Optional[float] = None) -> np.ndarray:
    """
    Computes invariant 80-dimensional feature vector from 21 MediaPipe hand landmarks.
    Features:
      - 63 wrist-relative coordinates normalized by palm length (21 * 3)
      - 7 key inter-fingertip distances
      - 5 fingertip-to-wrist extension distances
      - 5 finger curl / straightness angles
    """
    wrist = landmarks[0]
    mid_mcp = landmarks[9]

    if scale is None or scale < 1e-4:
        scale = math.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y)
        if scale < 1e-4:
            scale = 1.0

    features = []

    # 1. 63 Normalized joint positions relative to wrist
    for lm in landmarks:
        features.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])

    thb_tip = landmarks[4]
    idx_tip = landmarks[8]
    mid_tip = landmarks[12]
    rng_tip = landmarks[16]
    pnk_tip = landmarks[20]

    def dist3d(p1, p2):
        return math.sqrt(
            ((p1.x - p2.x) / scale) ** 2 +
            ((p1.y - p2.y) / scale) ** 2 +
            ((p1.z - p2.z) / scale) ** 2
        )

    # 2. Key inter-finger distances (7 features)
    features.append(dist3d(thb_tip, idx_tip))
    features.append(dist3d(thb_tip, mid_tip))
    features.append(dist3d(thb_tip, rng_tip))
    features.append(dist3d(thb_tip, pnk_tip))
    features.append(dist3d(idx_tip, mid_tip))
    features.append(dist3d(mid_tip, rng_tip))
    features.append(dist3d(rng_tip, pnk_tip))

    # 3. Fingertip-to-wrist extensions (5 features)
    features.append(dist3d(thb_tip, wrist))
    features.append(dist3d(idx_tip, wrist))
    features.append(dist3d(mid_tip, wrist))
    features.append(dist3d(rng_tip, wrist))
    features.append(dist3d(pnk_tip, wrist))

    # 4. Finger extension ratios relative to MCP (5 features)
    for mcp_idx, tip_idx in [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]:
        d_tip = dist3d(landmarks[tip_idx], wrist)
        d_mcp = dist3d(landmarks[mcp_idx], wrist)
        features.append(d_tip / (d_mcp + 1e-5))

    return np.array(features, dtype=np.float32)


def run_extraction(
    dataset_dir: Path,
    output_path: Path,
    samples_per_class: int = 150
) -> None:
    print("=" * 70)
    print("🖐️ Starting SignBridge AI 3D Landmark Feature Extraction")
    print("=" * 70)
    print(f"• Dataset Source:     {dataset_dir}")
    print(f"• Target Output:      {output_path}")
    print(f"• Samples per Class:  {samples_per_class}")

    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"• Found {len(classes)} classes: {classes}")

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.45
    )

    X_list = []
    y_list = []
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    start_time = time.time()
    total_images_processed = 0

    for cls_idx, cls_name in enumerate(classes):
        cls_dir = dataset_dir / cls_name
        img_files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png")))

        # Subsample evenly
        if len(img_files) > samples_per_class:
            indices = np.linspace(0, len(img_files) - 1, samples_per_class, dtype=int)
            selected_files = [img_files[i] for i in indices]
        else:
            selected_files = img_files

        cls_extracted = 0
        cls_start = time.time()

        for img_path in selected_files:
            total_images_processed += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if cls_name.lower() == "nothing":
                # For 'nothing' class, represent with neutral zeros
                feat = np.zeros(80, dtype=np.float32)
                X_list.append(feat)
                y_list.append(class_to_idx[cls_name])
                cls_extracted += 1
                continue

            results = hands_detector.process(rgb)
            if results and results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0].landmark
                feat = extract_features_from_landmarks(lms)
                X_list.append(feat)
                y_list.append(class_to_idx[cls_name])
                cls_extracted += 1

        cls_time = time.time() - cls_start
        print(f"[{cls_idx+1:02d}/{len(classes)}] Class '{cls_name}': {cls_extracted}/{len(selected_files)} hands extracted ({cls_time:.1f}s)")

    hands_detector.close()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        classes=np.array(classes)
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ Extraction Complete in {elapsed:.1f} seconds!")
    print(f"• Total Extracted Hand Samples: {len(X)}")
    print(f"• Feature Dimension:             {X.shape[1]}")
    print(f"• Saved to:                      {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    train_dir = CONFIG.dataset.train_dir
    if not train_dir.exists() or not any(train_dir.iterdir()):
        train_dir = PROJECT_ROOT / "dataset" / "asl_alphabet_train" / "asl_alphabet_train"

    out_file = PROJECT_ROOT / "dataset" / "landmarks_dataset.npz"
    run_extraction(dataset_dir=train_dir, output_path=out_file, samples_per_class=120)
