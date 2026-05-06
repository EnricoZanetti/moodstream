from pathlib import Path

import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

AFFECTNET_TEST_DIR = Path(__file__).parent / "AffectNet" / "Test"
MODEL_PATH = Path(__file__).parent / "emotion_model.tflite"

IMG_SIZE = 96
LABELS = ["happy", "sad", "angry", "neutral", "surprised", "fearful", "disgust"]
AFFECTNET_TO_OURS = {
    "happy": 0, "sad": 1, "anger": 2, "neutral": 3,
    "surprise": 4, "fear": 5, "disgust": 6,
}

images, labels_list = [], []
for class_dir in sorted(AFFECTNET_TEST_DIR.iterdir()):
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
        labels_list.append(label)

x_val, y_val = np.array(images), np.array(labels_list)

interp = Interpreter(model_path=str(MODEL_PATH))
interp.allocate_tensors()
inp, out = interp.get_input_details(), interp.get_output_details()

correct, per_class = 0, [[0, 0] for _ in LABELS]
for img, label in zip(x_val, y_val):
    interp.set_tensor(inp[0]["index"], img.reshape(1, IMG_SIZE, IMG_SIZE, 1))
    interp.invoke()
    pred = int(np.argmax(interp.get_tensor(out[0]["index"])[0]))
    per_class[label][1] += 1
    if pred == label:
        correct += 1
        per_class[label][0] += 1

print(f"Overall: {correct / len(y_val):.1%}  ({correct}/{len(y_val)})\n")
for i, name in enumerate(LABELS):
    c, t = per_class[i]
    print(f"  {name:12s}: {c / t:.1%}  ({c}/{t})")
