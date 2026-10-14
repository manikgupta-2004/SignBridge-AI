"""
SignBridge AI - Model Training Pipeline
Trains MobileNetV2 or EfficientNetB0 backbones with transfer learning and optional fine-tuning.
Saves model artifacts, class mappings, preprocessing configs, and training curves.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import CONFIG, PROJECT_ROOT
from src.data_loader import create_data_generators, discover_dataset_structure, resolve_training_directory
from src.model_builder import build_transfer_model, unfreeze_for_fine_tuning
from src.preprocessing import save_preprocessing_config


def train_model(
    model_type: str = "efficientnet_b0",
    dataset_dir: str = None,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    fine_tune: bool = False,
    fine_tune_epochs: int = 4,
    save_dir: str = None
) -> None:
    print("=" * 70)
    print("🧠 Starting SignBridge AI Model Training Pipeline")
    print("=" * 70)
    print(f"• Architecture:   {model_type}")
    print(f"• Batch Size:     {batch_size}")
    print(f"• Initial Epochs: {epochs}")
    print(f"• Learning Rate:  {learning_rate}")
    print(f"• Fine-tuning:    {fine_tune} (Epochs: {fine_tune_epochs if fine_tune else 0})")

    # Resolve paths
    resolved_train_dir = resolve_training_directory(Path(dataset_dir) if dataset_dir else None)
    out_dir = Path(save_dir) if save_dir else CONFIG.model.trained_dir
    checkpoint_dir = CONFIG.model.checkpoints_dir
    results_dir = CONFIG.results_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Inspect dataset
    print(f"\n[Dataset] Scanning directory: {resolved_train_dir}...")
    dataset_info = discover_dataset_structure(resolved_train_dir)
    print(f"• Found {dataset_info['num_classes']} classes: {dataset_info['classes'][:6]}... (+{dataset_info['num_classes']-6} more)")
    print(f"• Total Images:   {dataset_info['total_images']}")
    print(f"• Balanced:       {dataset_info['is_balanced']}")

    # Configure dataset generators
    CONFIG.dataset.train_dir = resolved_train_dir
    CONFIG.dataset.batch_size = batch_size
    train_gen, val_gen, class_indices, class_weights = create_data_generators(
        CONFIG.dataset,
        model_type=model_type
    )

    # Save class indices
    class_indices_path = out_dir / CONFIG.model.class_indices_file
    with open(class_indices_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)
    print(f"• Saved class indices mapping to {class_indices_path}")

    # Save preprocessing config
    prep_config = {
        "image_size": list(CONFIG.dataset.image_size),
        "model_type": model_type,
        "channels": 3,
        "color_mode": "RGB",
        "num_classes": len(class_indices)
    }
    prep_config_path = out_dir / CONFIG.model.preprocessing_config_file
    save_preprocessing_config(prep_config_path, prep_config)
    print(f"• Saved preprocessing config to {prep_config_path}")

    # Build model
    print(f"\n[Model] Building transfer learning model ({model_type})...")
    model = build_transfer_model(
        model_type=model_type,
        num_classes=len(class_indices),
        input_shape=(CONFIG.dataset.image_size[1], CONFIG.dataset.image_size[0], 3),
        dropout_rate=CONFIG.model.dropout_rate,
        dense_units=CONFIG.model.dense_units,
        learning_rate=learning_rate
    )
    model.summary(line_length=80)

    # Callbacks
    best_checkpoint_path = checkpoint_dir / "best_model.keras"
    callbacks = [
        ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Phase 1: Train transfer classification head
    print("\n[Phase 1] Training classification head with frozen backbone...")
    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights if not dataset_info["is_balanced"] else None,
        verbose=1
    )

    combined_history = {
        "accuracy": list(history_phase1.history.get("accuracy", [])),
        "val_accuracy": list(history_phase1.history.get("val_accuracy", [])),
        "loss": list(history_phase1.history.get("loss", [])),
        "val_loss": list(history_phase1.history.get("val_loss", []))
    }

    # Phase 2: Optional Fine-Tuning
    if fine_tune:
        print("\n[Phase 2] Unfreezing top backbone layers for fine-tuning...")
        model = unfreeze_for_fine_tuning(
            model=model,
            unfreeze_layers=25,
            learning_rate=CONFIG.model.fine_tune_lr
        )
        history_phase2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1
        )
        for k in combined_history:
            combined_history[k].extend(history_phase2.history.get(k, []))

    # Save final model
    target_model_name = f"signbridge_{model_type}.keras"
    final_model_path = out_dir / target_model_name
    model.save(str(final_model_path))
    print(f"\n✅ Training complete! Final model saved to: {final_model_path}")

    # Generate and save learning curves
    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(combined_history["accuracy"], label="Train Accuracy", color="#00C49F", linewidth=2)
    plt.plot(combined_history["val_accuracy"], label="Val Accuracy", color="#0088FE", linewidth=2)
    plt.title(f"{model_type.upper()} Accuracy Progression", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(combined_history["loss"], label="Train Loss", color="#FF8042", linewidth=2)
    plt.plot(combined_history["val_loss"], label="Val Loss", color="#FFBB28", linewidth=2)
    plt.title(f"{model_type.upper()} Loss Progression", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    curves_path = results_dir / f"training_curves_{model_type}.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"• Learning curve plots saved to: {curves_path}")


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Model Trainer")
    parser.add_argument("--model_type", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v2"], help="Model backbone")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=8, help="Number of head training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--fine_tune", action="store_true", help="Perform unfreezing fine-tuning phase")
    parser.add_argument("--fine_tune_epochs", type=int, default=4, help="Fine-tuning epoch count")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for trained models")
    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
