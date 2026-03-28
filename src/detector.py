"""Webcam-based emotion detector with MQTT publishing.

Captures frames from the laptop webcam, detects faces using a Haar cascade,
classifies emotions with a TFLite model, and publishes results to an MQTT broker.
The IoT pipeline (MQTT -> Node-Red -> InfluxDB -> Grafana) is unchanged.

Usage:
    uv run python src/detector.py              # webcam + MQTT
    uv run python src/detector.py --no-display  # headless (no OpenCV window)
    uv run python src/detector.py --demo        # random emotions, no webcam
    uv run python src/detector.py --demo --no-mqtt  # demo without MQTT
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt

EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised", "fearful"]

MODEL_PATH = Path(__file__).parent.parent / "models" / "emotion_model.tflite"
LABELS_PATH = Path(__file__).parent.parent / "models" / "labels.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_labels(path: Path) -> list[str]:
    """Load emotion labels from text file."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_tflite_model(path: Path):
    """Load TFLite model, trying tflite-runtime first, then full TensorFlow."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            log.error(
                "Neither tflite-runtime nor tensorflow is installed. "
                "Install one with: uv add tflite-runtime  OR  uv add tensorflow"
            )
            sys.exit(1)

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize face to 48x48, normalize, and reshape for model input."""
    face = cv2.resize(face_img, (48, 48))
    face = face.astype(np.float32) / 255.0
    return face.reshape(1, 48, 48, 1)


def classify_emotion(interpreter, face_input: np.ndarray) -> tuple[str, float]:
    """Run TFLite inference and return (label, confidence)."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], face_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    labels = load_labels(LABELS_PATH)
    idx = int(np.argmax(output))
    return labels[idx], float(output[idx])


def create_mqtt_client(
    broker: str,
    port: int,
    username: str | None,
    password: str | None,
    client_id: str,
) -> mqtt.Client:
    """Create and connect an MQTT client."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id)
    if username:
        client.username_pw_set(username, password)
    client.connect(broker, port, keepalive=60)
    return client


def run_demo(mqtt_client: mqtt.Client | None, topic: str) -> None:
    """Publish random emotions for pipeline testing (no hardware needed)."""
    log.info("Demo mode - publishing random emotions every 2 s")

    while True:
        try:
            emotion = random.choice(EMOTIONS)
            if mqtt_client:
                mqtt_client.publish(topic, emotion)
                mqtt_client.loop()
            log.info("Published: %s", emotion)
            time.sleep(2)
        except KeyboardInterrupt:
            break


def run_webcam(
    mqtt_client: mqtt.Client | None,
    topic: str,
    show_display: bool,
) -> None:
    """Capture webcam frames, detect faces, classify emotions, publish to MQTT."""
    if not MODEL_PATH.exists():
        log.error("Model not found at %s", MODEL_PATH)
        log.error("Train it with: uv run python models/train.py")
        log.error("See models/README.md for details.")
        sys.exit(1)

    interpreter = load_tflite_model(MODEL_PATH)
    log.info("Loaded TFLite model from %s", MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Could not open webcam. Check that a camera is connected.")
        sys.exit(1)

    log.info("Webcam opened - press Ctrl+C or 'q' to quit")
    last_publish = 0.0
    publish_interval = 2.0  # publish at most every 2 seconds (matches demo pace)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame, retrying...")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
            )

            for x, y, w, h in faces:
                face_roi = gray[y : y + h, x : x + w]
                face_input = preprocess_face(face_roi)
                emotion, confidence = classify_emotion(interpreter, face_input)

                # Publish at a throttled rate
                now = time.time()
                if now - last_publish >= publish_interval:
                    if mqtt_client:
                        mqtt_client.publish(topic, emotion)
                        mqtt_client.loop()
                    log.info("Detected: %s (%.0f%%)", emotion, confidence * 100)
                    last_publish = now

                if show_display:
                    label = f"{emotion} ({confidence:.0%})"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )

            if show_display:
                cv2.imshow("Emotion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if mqtt_client:
                mqtt_client.loop(timeout=0.0)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show_display:
            cv2.destroyAllWindows()
        log.info("Webcam released, exiting.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam emotion detector with MQTT publishing",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic emotions (no webcam needed)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display window (headless mode)",
    )
    parser.add_argument(
        "--no-mqtt",
        action="store_true",
        help="Skip MQTT connection (log to stdout only, useful with --demo)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    broker = os.getenv("MQTT_BROKER", "localhost")
    port = int(os.getenv("MQTT_PORT", "1883"))
    topic = os.getenv("MQTT_TOPIC", "ezan/emotion_detection")
    username = os.getenv("MQTT_USERNAME") or None
    password = os.getenv("MQTT_PASSWORD") or None
    client_id = os.getenv("MQTT_CLIENT_ID", "webcam-emotion")

    mqtt_client = None
    if not args.no_mqtt:
        try:
            mqtt_client = create_mqtt_client(broker, port, username, password, client_id)
            log.info("Connected to MQTT broker at %s:%d", broker, port)
        except Exception as e:
            if args.demo:
                log.warning(
                    "Could not connect to MQTT broker: %s (continuing in stdout-only mode)", e
                )
            else:
                log.error("Could not connect to MQTT broker: %s", e)
                sys.exit(1)

    if args.demo:
        run_demo(mqtt_client, topic)
    else:
        run_webcam(mqtt_client, topic, show_display=not args.no_display)


if __name__ == "__main__":
    main()
