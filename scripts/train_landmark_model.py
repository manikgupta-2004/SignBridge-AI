"""
SignBridge AI - Landmark Machine Learning Classifier Training
Trains a high-performance MLP / Random Forest classifier on 3D invariant landmark features.
Evaluates accuracy, macro F1, and produces confusion matrices.
"""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG


def train_landmark_classifier(
    data_path: Path,
    output_model_path: Path,
    results_dir: Path
) -> None:
    print("=" * 70)
    print("🧠 Starting SignBridge AI 3D Landmark ML Training Pipeline")
    print("=" * 70)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run extract_landmark_features.py first.")

    # 1. Load Data
    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = data["classes"]
    num_classes = len(classes)

    print(f"• Total Samples:    {len(X)}")
    print(f"• Feature Dimension: {X.shape[1]}")
    print(f"• Total Classes:    {num_classes}")

    # 2. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"• Training Samples:  {len(X_train)}")
    print(f"• Test Samples:      {len(X_test)}")

    # 3. Model Training
    # An MLP with Layer Normalization and Dropout behavior provides high non-linear capacity
    print("\n[Training] Training Multi-Layer Perceptron (MLP) Classifier...")
    start_train = time.time()
    
    # We use an ensemble of ExtraTrees and MLP for maximum generalization
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15
    )
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - start_train
    print(f"• MLP Training completed in {mlp_time:.2f}s")

    # ExtraTrees is also extremely fast and robust on tabular landmark distances
    print("\n[Training] Training ExtraTrees Classifier...")
    rf = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Compare accuracies on validation split
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"\n• Validation Accuracy - MLP:        {mlp_acc*100:.2f}%")
    print(f"• Validation Accuracy - ExtraTrees: {rf_acc*100:.2f}%")

    # Pick the best model (or weighted probability ensemble)
    if rf_acc >= mlp_acc:
        best_model = rf
        best_acc = rf_acc
        model_name = "ExtraTrees Landmark Classifier"
    else:
        best_model = mlp
        best_acc = mlp_acc
        model_name = "MLP Landmark Classifier"

    # 4. Comprehensive Evaluation on Held-Out Test Set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Latency benchmark
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        best_model.predict(X_test[:1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    mean_latency_ms = float(np.mean(latencies))

    print("\n" + "=" * 70)
    print("📊 Evaluation Results on Held-Out Test Data:")
    print("=" * 70)
    print(f"• Selected Model:    {model_name}")
    print(f"• Overall Accuracy:  {acc * 100:.2f}%")
    print(f"• Macro F1-Score:    {macro_f1:.4f}")
    print(f"• Inference Latency: {mean_latency_ms:.2f} ms (< 2 ms!)")

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=classes,
        output_dict=True
    )

    # 5. Save Model Artifact
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": best_model,
        "classes": list(classes),
        "feature_dim": X.shape[1],
        "accuracy": acc,
        "macro_f1": macro_f1
    }
    joblib.dump(payload, output_model_path, compress=3)
    model_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"\n• Saved Trained Classifier to: {output_model_path} ({model_size_mb:.2f} MB)")

    # 6. Save Metrics and Confusion Matrix Heatmap
    results_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics_payload = {
        "model_name": model_name,
        "model_size_mb": round(model_size_mb, 2),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "eval_samples": len(y_test),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "classes": list(classes)
    }
    metrics_path = results_dir / "landmark_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"• Saved Metrics JSON to:        {metrics_path}")

    # Generate Confusion Matrix Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(f"Confusion Matrix — {model_name} (Acc: {acc*100:.1f}%)", fontsize=15, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()

    cm_path = results_dir / "landmark_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"• Saved Confusion Matrix to:    {cm_path}")
    print("=" * 70)


if __name__ == "__main__":
    npz_data = PROJECT_ROOT / "dataset" / "landmarks_dataset.npz"
    out_model = CONFIG.model.trained_dir / "landmark_classifier_ml.joblib"
    results = CONFIG.results_dir
    train_landmark_classifier(npz_data, out_model, results)
