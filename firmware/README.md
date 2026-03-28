# Firmware - OpenMV Cam H7 Plus (Legacy)

> **Note:** This is the legacy v1/v2 firmware for the OpenMV Cam H7+. For the current webcam-based version (v3), see [`src/detector.py`](../src/detector.py).

This directory contains the MicroPython script that runs directly on the OpenMV Cam H7 Plus.

## Files Required on the Camera

| File | Description |
|------|-------------|
| `main.py` | Main script - auto-runs on boot |
| `trained.tflite` | TensorFlow Lite emotion detection model (generated via Edge Impulse) |
| `labels.txt` | One emotion label per line, matching model output order |

## Flashing Instructions

1. Connect the OpenMV Cam H7 Plus to your computer via USB.
2. Open **OpenMV IDE** and connect to the camera.
3. Copy `main.py` to the camera's internal flash storage.
4. Copy your `trained.tflite` model and `labels.txt` to the same location.
5. Reset the camera - `main.py` will run automatically on boot.

## How It Works

1. The camera captures frames in QVGA (320x240) RGB565 format.
2. A Haar cascade detector locates faces in each frame.
3. Each detected face region is classified by the TFLite model.
4. The predicted emotion label is drawn on the frame and sent via UART at 115200 baud.

The UART output is consumed by `host/mqtt_publisher.py` on the connected PC.
