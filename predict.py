"""
SignBridge AI - Single Image Prediction CLI
Runs inference on a single static image file or image folder and outputs top predicted ASL signs.
"""

import argparse
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG
from src.inference import SignPredictor


def run_prediction(
    image_path: str,
    model_path: str = None,
    visualize: bool = False,
    top_k: int = 3
) -> None:
    path = Path(image_path)
    if not path.exists():
        print(f"❌ Error: Image file '{image_path}' not found.")
        sys.exit(1)

    print("=" * 65)
    print(f"🔍 Analyzing image: {path.name}")
    print("=" * 65)

    predictor = SignPredictor(model_path=Path(model_path) if model_path else None)

    # Read image using OpenCV (BGR)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        print(f"❌ Error: Failed to read image file '{image_path}'. Ensure it is a valid JPEG/PNG.")
        sys.exit(1)

    # Predict
    result = predictor.predict_roi(img_bgr, is_bgr=True, top_k_num=top_k)

    print(f"\n🎯 Top Prediction:     {result.label.upper()}")
    print(f"✨ Confidence:         {result.confidence * 100:.2f}%")
    print(f"⚡ Inference Latency:  {result.latency_ms:.2f} ms")

    print("\nTop Candidate Predictions:")
    for rank, (cand_lbl, cand_conf) in enumerate(result.top_k, start=1):
        bar_len = int(cand_conf * 25)
        bar_str = "█" * bar_len + "░" * (25 - bar_len)
        print(f"  {rank}. [{cand_lbl.upper():^7}] {cand_conf*100:6.2f}%  |{bar_str}|")

    if visualize:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(
            f"Prediction: {result.label.upper()} ({result.confidence*100:.1f}%)\nLatency: {result.latency_ms:.1f}ms",
            fontsize=14,
            fontweight="bold",
            color="#0066CC" if result.confidence > 0.8 else "#CC3300"
        )
        plt.tight_layout()
        output_vis = path.parent / f"pred_{path.stem}.png"
        plt.savefig(output_vis, dpi=180)
        print(f"\n🖼️ Visualized prediction saved to: {output_vis}")
        try:
            plt.show()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Image Classifier")
    parser.add_argument("image", type=str, help="Path to input image file")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--visualize", action="store_true", help="Display/save visual prediction plot")
    parser.add_argument("--top_k", type=int, default=3, help="Number of candidate classes")
    args = parser.parse_args()

    run_prediction(
        image_path=args.image,
        model_path=args.model,
        visualize=args.visualize,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
