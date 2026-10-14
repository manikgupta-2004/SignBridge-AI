"""
SignBridge AI - Webcam Data Collection & Calibration Tool
Enables users to record custom hand gesture images using their webcam to bridge the domain shift
and fine-tune the model for specific lighting, background, and skin tones.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Add root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG
from src.preprocessing import extract_hand_roi, get_center_roi


def collect_webcam_data(
    target_class: str,
    output_dir: str = "dataset/custom",
    camera_index: int = 0,
    target_size: int = 224
) -> None:
    clean_class = target_class.strip()
    save_path = Path(output_dir) / clean_class
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"📸 SignBridge AI Webcam Calibration — Collecting Class '{clean_class.upper()}'")
    print("=" * 65)
    print(f"• Save Directory: {save_path}")
    print(f"• Camera Index:   {camera_index}")
    print("\nControls:")
    print("  [SPACE]  Capture 1 sample image")
    print("  [C]      Burst capture (record 20 frames automatically)")
    print("  [Q/ESC]  Finish collection and exit\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Cannot access camera {camera_index}.")
        return

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    mp_draw = mp.solutions.drawing_utils

    sample_count = len(list(save_path.glob("*.jpg")))
    burst_mode = False
    burst_remaining = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            roi = None
            bbox = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                roi, bbox = extract_hand_roi(frame, hand_landmarks, padding_ratio=0.28)

            if roi is None or bbox is None:
                roi, bbox = get_center_roi(frame, box_ratio=0.50)
                box_color = (0, 165, 255)
            else:
                box_color = (0, 255, 100)

            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

            # HUD Display
            cv2.putText(frame, f"CLASS: {clean_class.upper()}", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 215, 255), 2)
            cv2.putText(frame, f"Collected: {sample_count} images", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "[SPACE] Single | [C] Burst 20 | [Q] Quit", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Handle Burst capture
            if burst_mode and burst_remaining > 0:
                burst_remaining -= 1
                sample_count += 1
                resized_roi = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_AREA)
                filename = save_path / f"{clean_class}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(filename), resized_roi)
                cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)  # Red recording light
                if burst_remaining == 0:
                    burst_mode = False

            cv2.imshow("SignBridge AI - Dataset Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == 32:  # Single capture
                sample_count += 1
                resized_roi = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_AREA)
                filename = save_path / f"{clean_class}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(filename), resized_roi)
                print(f"📸 Captured sample #{sample_count} -> {filename.name}")
            elif key in (ord("c"), ord("C")):
                burst_mode = True
                burst_remaining = 20
                print(f"🚀 Starting 20-frame burst capture for class '{clean_class}'...")

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print(f"\n✅ Collection finished. Total images saved: {sample_count} in {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Webcam Gesture Data Collector")
    parser.add_argument("class_name", type=str, help="Target class name (e.g. 'A', 'B', 'space', 'nothing')")
    parser.add_argument("--output_dir", type=str, default="dataset/custom", help="Output directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    collect_webcam_data(
        target_class=args.class_name,
        output_dir=args.output_dir,
        camera_index=args.camera
    )


if __name__ == "__main__":
    main()
