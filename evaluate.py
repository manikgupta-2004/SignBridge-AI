"""
SignBridge AI - Model Evaluation & Benchmarking Suite
Evaluates trained models on held-out test data, computing precision, recall,
macro F1-score, confusion matrix, top confused sign pairs, model disk size,
and runtime inference latency. Generates an exhaustive report.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from config import CONFIG, PROJECT_ROOT
from src.data_loader import create_data_generators, resolve_training_directory
from src.inference import SignPredictor


def evaluate_model(
    model_path: str = None,
    dataset_dir: str = None,
    num_eval_samples: int = 2900,  # ~100 per class for rapid, robust statistical evaluation
    batch_size: int = 32
) -> None:
    print("=" * 75)
    print("📊 Starting SignBridge AI Model Evaluation Suite")
    print("=" * 75)

    results_dir = CONFIG.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model and Predictor
    predictor = SignPredictor(model_path=Path(model_path) if model_path else None)
    model = predictor.model
    class_names = predictor.class_names
    num_classes = len(class_names)

    # Calculate model file size
    model_file_path = predictor.model_path
    model_size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
    print(f"• Model File:       {model_file_path.name}")
    print(f"• Model Size:       {model_size_mb:.2f} MB")
    print(f"• Total Classes:    {num_classes}")

    # 2. Benchmark Inference Latency
    print("\n[Benchmark] Measuring inference latency (50 iterations)...")
    latency_stats = predictor.benchmark_latency(num_samples=50)
    print(f"• Mean Latency:     {latency_stats['mean_latency_ms']} ms")
    print(f"• 95th Percentile:  {latency_stats['p95_latency_ms']} ms")
    print(f"• Projected FPS:    {latency_stats['estimated_fps']} FPS")

    # 3. Prepare Evaluation Test Data Generator
    print("\n[Evaluation] Loading independent evaluation subset...")
    resolved_train_dir = resolve_training_directory(Path(dataset_dir) if dataset_dir else None)
    CONFIG.dataset.train_dir = resolved_train_dir
    CONFIG.dataset.batch_size = batch_size

    # Use validation generator as the held-out evaluation set
    _, eval_gen, _, _ = create_data_generators(CONFIG.dataset, model_type=predictor.model_type)

    total_eval_available = len(eval_gen.filenames)
    batches_to_evaluate = min(len(eval_gen), int(np.ceil(num_eval_samples / batch_size)))
    actual_samples = min(total_eval_available, batches_to_evaluate * batch_size)
    print(f"• Evaluating on {actual_samples} held-out validation images across {num_classes} classes...")

    # Strided batch sampling so samples are evenly distributed across all classes A-space
    total_batches = len(eval_gen)
    selected_batch_indices = np.linspace(0, total_batches - 1, batches_to_evaluate, dtype=int)

    # Collect Ground Truth and Predictions
    y_true_indices = []
    y_pred_indices = []

    start_eval_time = time.time()
    for step, b_idx in enumerate(selected_batch_indices, start=1):
        x_batch, y_batch = eval_gen[int(b_idx)]
        preds = model.predict(x_batch, verbose=0)
        
        batch_true = np.argmax(y_batch, axis=1)
        batch_pred = np.argmax(preds, axis=1)
        
        y_true_indices.extend(batch_true)
        y_pred_indices.extend(batch_pred)
        
        if step % 5 == 0 or step == len(selected_batch_indices):
            print(f"  Processed {len(y_true_indices)} / {actual_samples} samples...")

    eval_duration = time.time() - start_eval_time
    y_true = np.array(y_true_indices)
    y_pred = np.array(y_pred_indices)

    # 4. Compute Metrics
    all_class_indices = list(range(num_classes))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=all_class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=all_class_indices,
        target_names=class_names,
        zero_division=0
    )

    overall_accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]

    print("\n" + "=" * 75)
    print("🎯 Model Performance Summary")
    print("=" * 75)
    print(f"• Test Accuracy:    {overall_accuracy * 100:.2f}%")
    print(f"• Macro F1-Score:   {macro_f1:.4f}")
    print(f"• Macro Precision: {macro_precision:.4f}")
    print(f"• Macro Recall:    {macro_recall:.4f}")

    # 5. Compute Confusion Matrix & Top Errors
    cm = confusion_matrix(y_true, y_pred, labels=all_class_indices)

    # Identify most frequently confused pairs (excluding diagonal)
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], int(cm[i, j])))

    # Sort descending by error count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_5_confused = confused_pairs[:5]

    print("\n⚠️ Most Frequently Confused Sign Pairs:")
    if top_5_confused:
        for true_cls, pred_cls, err_cnt in top_5_confused:
            print(f"  • True '{true_cls}' misclassified as '{pred_cls}': {err_cnt} times")
    else:
        print("  • No systematic confusion observed on held-out split!")

    # 6. Plot & Save Confusion Matrix
    print("\n[Plots] Generating Confusion Matrix heatmap...")
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(f"Confusion Matrix — {model_file_path.name} (Acc: {overall_accuracy*100:.1f}%)", fontsize=15, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Ground Truth Class", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_plot_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"• Confusion matrix heatmap saved to: {cm_plot_path}")

    # 7. Save Machine-Readable Metrics JSON
    metrics_summary = {
        "model_name": model_file_path.name,
        "model_size_mb": round(model_size_mb, 2),
        "mean_latency_ms": latency_stats["mean_latency_ms"],
        "p95_latency_ms": latency_stats["p95_latency_ms"],
        "fps": latency_stats["estimated_fps"],
        "eval_samples": len(y_true),
        "accuracy": round(overall_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "top_confused_pairs": [
            {"true": t, "predicted": p, "count": c} for t, p, c in top_5_confused
        ]
    }
    metrics_json_path = results_dir / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"• Metrics JSON saved to: {metrics_json_path}")

    # 8. Write Comprehensive Evaluation Markdown Report
    report_md_path = results_dir / "evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# SignBridge AI — Model Evaluation & Diagnostic Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Evaluated Model:** `{model_file_path.name}` ({model_size_mb:.2f} MB)\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **Overall Accuracy** | **{overall_accuracy * 100:.2f}%** |\n")
        f.write(f"| **Macro F1-Score** | **{macro_f1:.4f}** |\n")
        f.write(f"| **Macro Precision** | **{macro_precision:.4f}** |\n")
        f.write(f"| **Macro Recall** | **{macro_recall:.4f}** |\n")
        f.write(f"| **Inference Latency (Mean)** | **{latency_stats['mean_latency_ms']} ms** |\n")
        f.write(f"| **Inference Throughput** | **~{latency_stats['estimated_fps']} FPS** |\n")
        f.write(f"| **Evaluation Test Samples** | **{len(y_true)}** |\n\n")

        f.write("## 2. Top Confused Sign Pairs\n\n")
        f.write("Certain ASL alphabet letters share identical finger arrangements with subtle wrist rotation or thumb placement differences:\n\n")
        f.write("| Ground Truth Sign | Mispredicted Sign | Misclassification Count |\n")
        f.write("| :--- | :--- | :--- |\n")
        for t, p, c in top_5_confused:
            f.write(f"| **{t}** | **{p}** | {c} |\n")
        f.write("\n")

        f.write("## 3. Full Per-Class Classification Report\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n\n")

        f.write("## 4. The Real-World Domain Shift (The Kaggle Gap)\n\n")
        f.write("> **Why 99%+ Dataset Accuracy Does NOT Guarantee 99% Webcam Accuracy:**\n\n")
        f.write("The Kaggle ASL Alphabet dataset consists of curated hand images captured under relatively uniform studio conditions with consistent camera distances, static backgrounds, and controlled lighting.\n\n")
        f.write("When transitioning to a live laptop webcam in the real world, several severe distribution shifts occur:\n")
        f.write("1. **Background Clutter & Hand Silhouette:** Webcams capture bedroom walls, lamps, curtains, and shirts. Without hand segmentation, edge filters can activate on background objects.\n")
        f.write("2. **Illumination & Shadow Variability:** Warm tungsten lighting, overhead fluorescent glare, or harsh side-lighting create shadows on finger joints not present in the dataset.\n")
        f.write("3. **Camera Angle & Focal Perspective:** Laptop webcams are often tilted upward at the user's chin. Finger foreshortening alters relative landmark positions.\n")
        f.write("4. **Skin Tone & Morphology Variations:** Hand sizes, finger thicknesses, and wrist angles vary across individuals.\n\n")
        f.write("**Solution Implemented in SignBridge AI:**\n")
        f.write("- MediaPipe Hand Detection isolates and crops only the hand region, normalizing background variance.\n")
        f.write("- Temporal smoothing filters out single-frame confidence dips.\n")
        f.write("- Calibration tool (`scripts/collect_webcam_data.py`) allows instant collection of personal calibration images for fine-tuning.\n")

    print(f"• Evaluation report saved to: {report_md_path}")
    print("\n✅ Evaluation successfully completed!")


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Model Evaluator")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model file")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--samples", type=int, default=2900, help="Number of held-out evaluation samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        dataset_dir=args.dataset_dir,
        num_eval_samples=args.samples,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
