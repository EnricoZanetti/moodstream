# Moodstream Model

This directory contains the TFLite model used by `src/detector.py` for real-time emotion classification.

## Files

| File | Description |
|------|-------------|
| `emotion_model.tflite` | Quantized TFLite CNN for 6-class emotion detection |
| `labels.txt` | Emotion labels in model output order |
| `train.py` | Training script to reproduce or retrain the model |

## Model Details

- **Architecture:** Small CNN (3 conv blocks + dense layers)
- **Input:** 48x48 grayscale image, normalized to [0, 1]
- **Output:** 6-class softmax (happy, sad, angry, neutral, surprised, fearful)
- **Dataset:** [FER2013](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data) (~30,000 images, Disgust class excluded)
- **Format:** TensorFlow Lite (quantized float16 for smaller size)

## Training

To retrain the model from scratch:

```bash
# Download FER2013 dataset from Kaggle first
# Place fer2013.csv in this directory, then:
uv run python models/train.py
```

This will:
1. Load and preprocess FER2013 data
2. Train a CNN for 30 epochs with early stopping
3. Export `emotion_model.tflite` to this directory

## Swapping the Model

You can replace `emotion_model.tflite` with any TFLite model that:
1. Accepts a 48x48 grayscale input (shape: `[1, 48, 48, 1]`, dtype: `float32`)
2. Outputs a probability vector matching the labels in `labels.txt`

Update `labels.txt` if your model uses different classes or a different order.
