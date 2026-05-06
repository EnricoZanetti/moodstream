"""Train a small CNN on AffectNet and export as TFLite.

Usage:
    uv run python models/train.py

Requires the AffectNet/ directory in models/ with Train/ and Test/ subdirectories.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_DIR = Path(__file__).parent
AFFECTNET_DIR = MODEL_DIR / "AffectNet"
OUTPUT_PATH = MODEL_DIR / "emotion_model.tflite"

IMG_SIZE = 96
NUM_CLASSES = 7

# Map AffectNet folder names (lowercased) to our class indices
# contempt is skipped (not in this dict)
AFFECTNET_TO_OURS = {
    "happy": 0,
    "sad": 1,
    "anger": 2,
    "neutral": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
}


def load_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all images from an AffectNet split directory (Train or Test)."""
    images, labels = [], []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        if class_name not in AFFECTNET_TO_OURS:
            continue
        label = AFFECTNET_TO_OURS[class_name]

        for img_path in class_dir.iterdir():
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            images.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
            labels.append(label)

    return np.array(images), np.array(labels)


def build_model():
    """Build a CNN for 96x96 grayscale emotion classification (7 classes)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
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
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
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
    train_dir = AFFECTNET_DIR / "Train"
    test_dir = AFFECTNET_DIR / "Test"

    if not train_dir.exists() or not test_dir.exists():
        print(f"Error: AffectNet dataset not found at {AFFECTNET_DIR}")
        print("Expected structure: AffectNet/Train/<class>/ and AffectNet/Test/<class>/")
        raise SystemExit(1)

    print("Loading AffectNet train set...")
    x_train, y_train = load_split(train_dir)
    print(f"Train: {len(x_train)} samples")

    print("Loading AffectNet test set...")
    x_val, y_val = load_split(test_dir)
    print(f"Val:   {len(x_val)} samples")

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
