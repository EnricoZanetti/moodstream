# MoodStream

<a href="https://www.python.org/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12"></a>
<a href="LICENSE" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
<img src="https://img.shields.io/badge/Version-v3.0.2-blueviolet.svg" alt="Version: v3.0.2">

Real-time facial emotion detection using your **laptop webcam** and a TFLite CNN model. Detected emotions flow through a full IoT pipeline - MQTT to Node-Red to InfluxDB - and are visualized in a live Grafana dashboard.

> Read blog post here: [Introducing MoodStream](https://enricozanetti.dev/blog/introducing-moodstream/)

![Demo](images/moodstream-demo-gif.gif)

> **v3** replaces the OpenMV Cam H7+ hardware requirement with webcam-based detection. Clone, `make run`, and see emotions flowing through the pipeline. The pre-trained model is included — no training required. The [legacy OpenMV firmware](firmware/) is still available for hardware users.

## Architecture

<p align="center">
  <img src="images/embedded_emotion_nn_pipeline.svg" alt="IoT Pipeline" width="700">
</p>

## Key Results

| Metric | Value |
|--------|-------|
| Classes | Happy, Sad, Angry, Neutral, Surprised, Fearful |
| Input | 48x48 grayscale face images |
| Dataset | FER2013 (~30,000 images) |
| Model | Small CNN exported as TFLite (see [`models/`](models/)) |
| Inference | Real-time on CPU via laptop webcam |

## Project Structure

```
├── src/
│   └── detector.py             # Webcam capture + face detection + emotion classification + MQTT
├── models/
│   ├── emotion_model.tflite    # Pre-trained TFLite model (ready to use)
│   ├── labels.txt              # Emotion labels in model output order
│   ├── train.py                # Training script (FER2013 → TFLite) — optional, for retraining
│   └── README.md               # Model documentation + how to swap
├── firmware/                   # Legacy OpenMV Cam H7+ code (v1/v2)
│   ├── main.py
│   └── README.md
├── host/
│   └── mqtt_publisher.py       # UART-to-MQTT bridge (for OpenMV hardware users)
├── pipeline/                   # IoT infrastructure (Mosquitto, Node-Red, InfluxDB, Grafana)
│   └── node-red-flows.json
├── notebooks/
│   └── dataset_analysis.ipynb
├── images/                     # Diagrams, screenshots, demo GIF
├── report/                     # Project report (PDF)
├── docker-compose.yml          # One-command IoT pipeline setup
├── Makefile                    # Development commands
├── .env.example                # MQTT configuration template
├── pyproject.toml              # Dependencies (managed by uv)
└── uv.lock                    # Locked dependency versions
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/EnricoZanetti/moodstream.git
cd moodstream
make setup
cp .env.example .env   # edit if needed
```

### 2. Try the Demo (No Webcam Needed)

```bash
make pipeline-up    # start Mosquitto, Node-Red, InfluxDB, Grafana
make demo           # publish random emotions every 2s
```

### 3. Run with Webcam

```bash
make pipeline-up
make run            # opens webcam, detects faces, classifies + publishes emotions
```

| Command | Description |
|---------|-------------|
| `make run` | Webcam + MQTT + display window |
| `make run-headless` | Webcam + MQTT, no display (e.g. for SSH) |
| `make demo` | Random emotions to MQTT (no webcam) |
| `make demo-local` | Random emotions to stdout only |

### Pipeline Dashboards

| Service | URL | Description |
|---------|-----|-------------|
| **Grafana** | <a href="http://localhost:3000" target="_blank" rel="noopener noreferrer">localhost:3000</a> | Emotion detection dashboard (no login required) |
| **Node-Red** | <a href="http://localhost:1880" target="_blank" rel="noopener noreferrer">localhost:1880</a> | Flow editor - inspect MQTT → InfluxDB pipeline |
| **InfluxDB** | <a href="http://localhost:8086" target="_blank" rel="noopener noreferrer">localhost:8086</a> | Time-series database UI (login: `admin` / `adminpassword`) |
| **Mosquitto** | `localhost:1883` | MQTT broker (use an MQTT client to inspect) |

<p align="center">
  <img src="images/grafana-v2-dashboard.png" alt="Grafana Dashboard" width="800">
</p>

## Legacy Hardware Setup (OpenMV Cam H7+)

<details>
<summary>Click to expand</summary>

The v1/v2 architecture used an OpenMV Cam H7+ for on-device inference, sending emotions over UART to a host PC running `host/mqtt_publisher.py`.

#### Requirements

- OpenMV Cam H7 Plus + <a href="https://openmv.io/pages/download" target="_blank" rel="noopener noreferrer">OpenMV IDE</a>
- Python 3.12+, <a href="https://docs.astral.sh/uv/" target="_blank" rel="noopener noreferrer">uv</a>, Docker & Docker Compose

#### Steps

1. Train a model via <a href="https://www.edgeimpulse.com/" target="_blank" rel="noopener noreferrer">Edge Impulse</a> and export as `trained.tflite` + `labels.txt`
2. Flash the camera - see [firmware/README.md](firmware/README.md)
3. `make pipeline-up`
4. `python host/mqtt_publisher.py`
5. View dashboard at <a href="http://localhost:3000" target="_blank" rel="noopener noreferrer">localhost:3000</a>

</details>

## Dataset

The <a href="https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data" target="_blank" rel="noopener noreferrer">FER2013</a> dataset contains ~35,000 grayscale 48x48 facial expression images. The Disgust class was discarded due to insufficient samples, leaving 6 classes balanced to ~30,000 images.

Explore the dataset distribution in [`notebooks/dataset_analysis.ipynb`](notebooks/dataset_analysis.ipynb).

## Challenges and Limitations

- **Model accuracy** - FER2013 is small and noisy; accuracy is limited compared to larger datasets
- **Lighting & angle** - webcam detection is sensitive to lighting conditions and face angle
- **Single face focus** - publishes the first detected face's emotion per interval

## Future Improvements

- Train with larger/cleaner datasets for higher accuracy
- Add TLS encryption for secure MQTT communication
- Integrate additional sensors (e.g., heart rate) for multimodal emotion analysis
- Explore real-time model fine-tuning based on user feedback

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

- <a href="https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data" target="_blank" rel="noopener noreferrer">FER2013 Dataset</a> - Kaggle
- <a href="https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html" target="_blank" rel="noopener noreferrer">OpenCV Haar Cascades</a> - Face detection
- <a href="https://www.tensorflow.org/lite" target="_blank" rel="noopener noreferrer">TensorFlow Lite</a> - On-device ML inference
- <a href="https://mosquitto.org/" target="_blank" rel="noopener noreferrer">Eclipse Mosquitto</a> - MQTT broker
- <a href="https://nodered.org/" target="_blank" rel="noopener noreferrer">Node-Red</a> - Flow-based IoT programming
- <a href="https://www.influxdata.com/" target="_blank" rel="noopener noreferrer">InfluxDB</a> - Time-series database
- <a href="https://grafana.com/" target="_blank" rel="noopener noreferrer">Grafana</a> - Data visualization
