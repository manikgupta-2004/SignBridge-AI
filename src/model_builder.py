"""
SignBridge AI - Model Builder Module
Constructs, configures, compiles, and fine-tunes transfer learning models for ASL recognition.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_transfer_model(
    model_type: str = "efficientnet_b0",
    num_classes: int = 29,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.35,
    dense_units: int = 512,
    learning_rate: float = 1e-4
) -> models.Model:
    """
    Constructs a transfer-learning convolutional neural network.

    Args:
        model_type: Backbone architecture ('efficientnet_b0' or 'mobilenet_v2').
        num_classes: Number of target sign categories (e.g., 29).
        input_shape: Dimensions of the input tensor (H, W, C).
        dropout_rate: Regularization dropout probability.
        dense_units: Units in the intermediate dense feature layer.
        learning_rate: Optimizer initial learning rate.

    Returns:
        Compiled Keras Model instance ready for training.
    """
    model_type_clean = model_type.lower()

    if "mobilenet" in model_type_clean:
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
    elif "efficientnet" in model_type_clean:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported model architecture '{model_type}'. Choose 'efficientnet_b0' or 'mobilenet_v2'.")

    # Freeze feature extraction backbone for initial transfer learning
    base_model.trainable = False

    # Build classification head
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout1")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"SignBridge_{model_type}")

    # Compile model with standard metrics
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
        ]
    )

    return model


def unfreeze_for_fine_tuning(
    model: models.Model,
    unfreeze_layers: int = 30,
    learning_rate: float = 1e-5
) -> models.Model:
    """
    Unfreezes the top layers of the backbone for fine-tuning.

    Args:
        model: Compiled transfer learning model.
        unfreeze_layers: Number of top backbone layers to unfreeze.
        learning_rate: Reduced learning rate for fine-tuning.

    Returns:
        Re-compiled model with unfrozen top layers.
    """
    # Locate backbone layer
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        # Fallback to outer layers
        backbone = model

    backbone.trainable = True

    # Freeze all layers except the last `unfreeze_layers`
    if unfreeze_layers > 0 and len(backbone.layers) > unfreeze_layers:
        for layer in backbone.layers[:-unfreeze_layers]:
            layer.trainable = False
        for layer in backbone.layers[-unfreeze_layers:]:
            # Keep BatchNormalization frozen to prevent mean/var degradation
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    # Recompile with small learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
        ]
    )

    return model


def build_landmark_classifier(
    num_classes: int = 29,
    input_dim: int = 63,
    learning_rate: float = 1e-3
) -> models.Model:
    """
    Builds a lightweight Multi-Layer Perceptron (MLP) for MediaPipe 3D coordinates (21 * 3 = 63).

    Args:
        num_classes: Number of sign classes.
        input_dim: Coordinate feature vector length.
        learning_rate: Initial learning rate.

    Returns:
        Compiled MLP Keras model.
    """
    inputs = layers.Input(shape=(input_dim,), name="landmark_features")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SignBridge_LandmarkMLP")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
