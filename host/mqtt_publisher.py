"""MQTT publisher that bridges OpenMV Cam emotion data to an MQTT broker.

Reads emotion classifications from the OpenMV Cam via UART (serial) and
publishes them to a configurable MQTT broker. Supports a --demo mode that
generates synthetic emotions for testing the pipeline without hardware.
"""

import argparse
import logging
import os
import random
import sys
import time

import paho.mqtt.client as mqtt
import serial
import serial.tools.list_ports

EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised", "fearful"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def find_openmv_port() -> str | None:
    """Auto-detect the OpenMV Cam serial port (cross-platform)."""
    for port in serial.tools.list_ports.comports():
        if "OpenMV" in (port.description or "") or "usbmodem" in (port.device or ""):
            return port.device
    return None


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


def run_serial(mqtt_client: mqtt.Client, topic: str, port: str) -> None:
    """Read emotions from serial and publish to MQTT."""
    ser = serial.Serial(port, 115200, timeout=1)
    log.info("Listening on %s ...", port)

    while True:
        try:
            line = ser.readline().decode("utf-8").strip()
            if line:
                mqtt_client.publish(topic, line)
                log.info("Published: %s", line)
            mqtt_client.loop()
        except serial.SerialException as e:
            log.error("Serial error: %s", e)
            break
        except KeyboardInterrupt:
            break
        time.sleep(0.1)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge OpenMV emotion data to MQTT",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic emotions (no camera needed)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port override (e.g. /dev/ttyUSB0, COM3)",
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
    topic = os.getenv("MQTT_TOPIC", "ezan/moodstream")
    username = os.getenv("MQTT_USERNAME") or None
    password = os.getenv("MQTT_PASSWORD") or None
    client_id = os.getenv("MQTT_CLIENT_ID", "moodstream")

    mqtt_client = None
    if not args.no_mqtt:
        try:
            mqtt_client = create_mqtt_client(
                broker, port, username, password, client_id
            )
            log.info("Connected to MQTT broker at %s:%d", broker, port)
        except Exception as e:
            if args.demo:
                log.warning(
                    "Could not connect to MQTT broker: %s (continuing in stdout-only mode)",
                    e,
                )
            else:
                log.error("Could not connect to MQTT broker: %s", e)
                sys.exit(1)

    if args.demo:
        run_demo(mqtt_client, topic)
    else:
        serial_port = args.port or find_openmv_port()
        if not serial_port:
            log.error(
                "No OpenMV camera found. Use --port to specify manually, or --demo for testing."
            )
            sys.exit(1)
        log.info("Using serial port: %s", serial_port)
        run_serial(mqtt_client, topic, serial_port)


if __name__ == "__main__":
    main()
