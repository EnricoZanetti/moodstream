"""Train a small CNN on FER2013 and export as TFLite.

Usage:
    uv run python models/train.py

Requires fer2013.csv in the models/ directory (download from Kaggle).
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_DIR = Path(__file__).parent
FER2013_PATH = MODEL_DIR / "fer2013.csv"
OUTPUT_PATH = MODEL_DIR / "emotion_model.tflite"

# FER2013 emotion mapping - we drop class 1 (Disgust) due to low sample count
# Original: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# Ours:     0=Happy, 1=Sad, 2=Angry, 3=Neutral, 4=Surprised, 5=Fearful
FER_TO_OURS = {3: 0, 4: 1, 0: 2, 6: 3, 5: 4, 2: 5}


def load_fer2013() -> tuple[np.ndarray, np.ndarray]:
    """Load FER2013 dataset from CSV, dropping Disgust class."""
    import csv

    pixels_list = []
    labels_list = []

    with open(FER2013_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fer_label = int(row["emotion"])
            if fer_label not in FER_TO_OURS:
                continue  # skip Disgust
            pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.float32).reshape(48, 48, 1)
            pixels /= 255.0
            pixels_list.append(pixels)
            labels_list.append(FER_TO_OURS[fer_label])

    return np.array(pixels_list), np.array(labels_list)


def build_model():
    """Build a small CNN for 48x48 grayscale emotion classification."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(48, 48, 1)),
            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            # Dense
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(6, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def convert_to_tflite(model) -> bytes:
    """Convert Keras model to quantized TFLite."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def main() -> None:
    if not FER2013_PATH.exists():
        print(f"Error: {FER2013_PATH} not found.")
        print("Download fer2013.csv from: https://www.kaggle.com/datasets/deadskull7/fer2013")
        print(f"Place it in: {MODEL_DIR}/")
        raise SystemExit(1)

    print("Loading FER2013 dataset...")
    x, y = load_fer2013()
    print(f"Loaded {len(x)} samples across 6 classes")

    # Split: 80% train, 20% validation
    split = int(len(x) * 0.8)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ],
    )

    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"Validation accuracy: {val_acc:.1%}")

    print("Converting to TFLite...")
    tflite_model = convert_to_tflite(model)

    OUTPUT_PATH.write_bytes(tflite_model)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Saved {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
