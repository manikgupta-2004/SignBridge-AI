"""
SignBridge AI - Real-Time Webcam Application
High-framerate OpenCV desktop recognition interface with MediaPipe hand tracking,
HUD overlay, prediction stabilization, and debounced sentence construction.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import CONFIG, WebcamConfig
from src.inference import SignPredictor
from src.preprocessing import extract_hand_roi, get_center_roi
from src.landmark_classifier import LandmarkSignClassifier
from src.prediction_smoother import PredictionSmoother
from src.sentence_builder import SentenceBuilder
from src.tts_engine import TTSEngine


class WebcamRecognitionApp:
    """
    Real-time webcam inference loop featuring an augmented reality HUD.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        camera_index: int = 0,
        confidence_threshold: float = 0.78,
        smoothing_window: int = 9,
        hold_frames: int = 12
    ):
        print("=" * 65)
        print("🚀 Initializing SignBridge AI Real-Time Recognition...")
        print("=" * 65)

        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.show_landmarks = True

        # Initialize Inference Predictors (CNN + 3D Geometric Landmark Classifier)
        try:
            self.predictor = SignPredictor(model_path=model_path)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)

        self.landmark_classifier = LandmarkSignClassifier()

        # Initialize Stabilizer and Sentence Builder
        self.smoother = PredictionSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
            stability_ratio=0.65
        )
        self.sentence_builder = SentenceBuilder(
            hold_duration_frames=hold_frames,
            cooldown_frames=14,
            auto_accept_on_space=True
        )
        self.tts = TTSEngine()

        # MediaPipe Hands detector
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def run(self) -> None:
        """Launches the continuous video capture and prediction loop."""
        print(f"[Webcam] Opening camera device index {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)

        # Verify camera access
        if not cap.isOpened():
            print(f"\n❌ Camera Error: Could not open camera at index {self.camera_index}.")
            print("Troubleshooting tips:")
            print("  1. Check if macOS camera permission is granted to Terminal / VS Code / IDE.")
            print("     Go to System Settings -> Privacy & Security -> Camera.")
            print("  2. If using an external webcam, try camera index 1: 'python -m src.webcam --camera 1'")
            print("  3. Ensure no other application (Zoom, FaceTime, Photo Booth) is locking the webcam.\n")
            return

        # Configure camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.webcam.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.webcam.frame_height)

        window_name = "SignBridge AI - Real-Time ASL Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1100, 720)

        prev_time = time.time()
        fps = 0.0

        print("✅ Webcam opened successfully!")
        print("\nKeyboard Controls:")
        print("  [Q / ESC] Quit application")
        print("  [C]       Clear recognized sentence")
        print("  [U]       Undo last character")
        print("  [S]       Speak sentence via Text-to-Speech")
        print("  [L]       Toggle hand landmarks overlay")
        print("  [SPACE]   Insert manual space\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Frame acquisition failed. Exiting loop.")
                    break

                # Mirror frame horizontally for natural user experience
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Convert to RGB for MediaPipe processing
                clean_frame = frame.copy()
                rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                results = self.hands_detector.process(rgb_frame)

                detected_roi = None
                bbox = None

                # Hand detection branch
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Extract clean ROI from untouched frame first
                    detected_roi, bbox = extract_hand_roi(clean_frame, hand_landmarks, padding_ratio=0.15)

                    if self.show_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                # Fallback to centered guide box if no hand detected
                if detected_roi is None or bbox is None:
                    detected_roi, bbox = get_center_roi(clean_frame, box_ratio=0.50)
                    is_hand_detected = False
                else:
                    is_hand_detected = True

                # Draw bounding box on frame
                x_min, y_min, x_max, y_max = bbox
                box_color = (0, 220, 100) if is_hand_detected else (90, 90, 180)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

                # Run inference (Hybrid: 3D Landmarks with Whole-Word Phrases + CNN)
                raw_label = "Uncertain"
                raw_conf = 0.0

                if is_hand_detected and hand_landmarks:
                    lm_lbl, lm_cnf, _ = self.landmark_classifier.classify_landmarks(hand_landmarks, phrase_mode=True)
                    if lm_cnf >= 0.85:
                        raw_label = lm_lbl
                        raw_conf = lm_cnf

                if raw_label == "Uncertain" and detected_roi is not None and detected_roi.size > 0:
                    try:
                        pred_res = self.predictor.predict_roi(detected_roi, is_bgr=True)
                        raw_label = pred_res.label
                        raw_conf = pred_res.confidence
                    except Exception as err:
                        pass

                # Stabilize predictions through rolling window
                stable_label, stable_conf, is_stable = self.smoother.update(
                    raw_label if is_hand_detected else "nothing",
                    raw_conf if is_hand_detected else 0.0
                )

                # Update Sentence Builder
                committed_token = self.sentence_builder.update(stable_label, is_stable)

                # Calculate instantaneous FPS
                current_time = time.time()
                time_diff = current_time - prev_time
                if time_diff > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
                prev_time = current_time

                # Draw modern Heads-Up Display (HUD)
                self._draw_hud(
                    frame=frame,
                    fps=fps,
                    raw_label=raw_label,
                    raw_conf=raw_conf,
                    stable_label=stable_label,
                    stable_conf=stable_conf,
                    is_stable=is_stable,
                    is_hand_detected=is_hand_detected
                )

                cv2.imshow(window_name, frame)

                # Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                    break
                elif key in (ord("c"), ord("C")):
                    self.sentence_builder.clear()
                elif key in (ord("u"), ord("U")):
                    self.sentence_builder.undo()
                elif key in (ord("s"), ord("S")):
                    sentence = self.sentence_builder.get_text()
                    if sentence:
                        print(f"[TTS] Speaking: '{sentence}'")
                        self.tts.speak(sentence)
                elif key in (ord("1"), ord("2"), ord("3")):
                    idx = int(chr(key)) - 1
                    suggs = self.sentence_builder.get_suggestions()
                    if idx < len(suggs):
                        self.sentence_builder.accept_suggestion(suggs[idx])
                elif key in (ord("l"), ord("L")):
                    self.show_landmarks = not self.show_landmarks
                elif key == 32:  # Spacebar
                    self.sentence_builder._commit_token("space")

        finally:
            print("\n[Webcam] Cleaning up resources...")
            cap.release()
            self.hands_detector.close()
            cv2.destroyAllWindows()
            print("✅ Resources successfully released. Goodbye!")

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        raw_label: str,
        raw_conf: float,
        stable_label: str,
        stable_conf: float,
        is_stable: bool,
        is_hand_detected: bool
    ) -> None:
        """Renders transparent glassmorphic overlay panels and telemetry onto frame."""
        h, w = frame.shape[:2]

        # Top telemetry bar overlay
        top_bar = frame.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 85), (20, 24, 30), -1)
        cv2.addWeighted(top_bar, 0.75, frame, 0.25, 0, frame)

        # Title and Branding
        cv2.putText(frame, "SIGNBRIDGE AI", (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 215, 255), 2)
        cv2.putText(frame, "Real-Time ASL Alphabet Recognition", (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 190, 200), 1)

        # FPS & Tracking status
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (w - 140, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

        hand_status = "HAND TRACKED" if is_hand_detected else "SEARCHING..."
        status_color = (0, 230, 100) if is_hand_detected else (100, 140, 255)
        cv2.putText(frame, hand_status, (w - 200, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        # Right prediction card
        card_w, card_h = 240, 200
        card_x = w - card_w - 20
        card_y = 105

        card_overlay = frame.copy()
        cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 30, 38), -1)
        cv2.addWeighted(card_overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (50, 60, 75), 1)

        # Display predicted class name
        cv2.putText(frame, "DETECTED SIGN", (card_x + 15, card_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 175, 190), 1)

        display_char = stable_label.upper() if is_stable else "?"
        char_color = (0, 255, 150) if is_stable else (120, 130, 140)
        if len(display_char) > 2:
            cv2.putText(frame, display_char, (card_x + 15, card_y + 85), cv2.FONT_HERSHEY_DUPLEX, 0.72, char_color, 2)
        else:
            cv2.putText(frame, display_char, (card_x + 85, card_y + 95), cv2.FONT_HERSHEY_DUPLEX, 1.8, char_color, 3)

        # Confidence percentage
        conf_pct = int(stable_conf * 100) if is_stable else int(raw_conf * 100)
        cv2.putText(frame, f"Confidence: {conf_pct}%", (card_x + 15, card_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Progress bar for character hold time
        hold_ratio = self.sentence_builder.get_progress_ratio()
        bar_x, bar_y = card_x + 15, card_y + 145
        bar_w, bar_h = card_w - 30, 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 75), -1)
        fill_w = int(bar_w * hold_ratio)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 215, 255), -1)

        # Stability badge
        stability_str = "STABLE" if is_stable else "UNCERTAIN"
        badge_color = (0, 200, 100) if is_stable else (80, 100, 230)
        cv2.putText(frame, stability_str, (card_x + 15, card_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_color, 1)

        # Bottom sentence panel
        bottom_h = 100
        bottom_y = h - bottom_h
        bot_overlay = frame.copy()
        cv2.rectangle(bot_overlay, (0, bottom_y), (w, h), (18, 22, 28), -1)
        cv2.addWeighted(bot_overlay, 0.82, frame, 0.18, 0, frame)
        cv2.line(frame, (0, bottom_y), (w, bottom_y), (40, 50, 65), 1)

        cv2.putText(frame, "RECOGNIZED SENTENCE:", (25, bottom_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

        sentence = self.sentence_builder.get_text()
        display_sentence = sentence[-45:] if len(sentence) > 45 else sentence
        cv2.putText(frame, f'"{display_sentence}"', (25, bottom_y + 65), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Autocomplete suggestions display
        suggs = self.sentence_builder.get_suggestions()
        if suggs:
            sugg_str = "   ".join([f"[{i+1}] {s}" for i, s in enumerate(suggs[:3])])
            cv2.putText(frame, f"Suggestions: {sugg_str}", (25, bottom_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 240, 160), 1)

        # Keyboard shortcuts footer
        shortcuts = "[1-3] Accept  |  [Space] Sugg/Word  |  [C] Clear  |  [S] Speak"
        cv2.putText(frame, shortcuts, (w - 510, bottom_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 165), 1)


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Real-Time Webcam Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .keras model file")
    parser.add_argument("--threshold", type=float, default=0.78, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--window", type=int, default=9, help="Smoothing window frame count")
    parser.add_argument("--hold", type=int, default=12, help="Frames to hold sign before commit")
    args = parser.parse_args()

    app = WebcamRecognitionApp(
        model_path=Path(args.model) if args.model else None,
        camera_index=args.camera,
        confidence_threshold=args.threshold,
        smoothing_window=args.window,
        hold_frames=args.hold
    )
    app.run()


if __name__ == "__main__":
    main()
