"""
This script is designed to run on a PC to read emotion data from an OpenMV Cam H7 Plus via UART and
publish the detected emotions to an MQTT broker. The script performs the following tasks:

1. Lists available serial ports and selects the correct one for the OpenMV camera.
2. Initializes an MQTT client and connects to the specified MQTT broker.
3. Continuously reads emotion data from the serial port.
4. Publishes the detected emotions to the MQTT broker under the topic "ezan/emotion_detection".

Key functionalities:
- Serial communication with the OpenMV camera.
- MQTT communication to publish emotion data.
"""

import paho.mqtt.client as mqtt
import serial
import time
import os

# Function to list available serial ports
def list_serial_ports():
    ports = [port for port in os.listdir('/dev') if 'tty.usbmodem' in port]
    return ['/dev/' + port for port in ports]

# Identify available serial ports
available_ports = list_serial_ports()
print(f"Available serial ports: {available_ports}")

# Select the correct serial port (assuming only one OpenMV camera is connected)
if available_ports:
    serial_port = available_ports[0]
else:
    raise Exception("No serial ports found. Ensure your OpenMV camera is connected.")

print(f"Using serial port: {serial_port}")

# MQTT settings
MQTT_BROKER = "x.x.x.x"
MQTT_PORT = yyyyy
MQTT_TOPIC = "your_topic"
MQTT_USERNAME = "your_username"
MQTT_PASSWORD = "your_password"
MQTT_CLIENT_ID = "your_client_id"

# Initialize MQTT client with the correct callback API version
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Initialize serial connection
ser = serial.Serial(serial_port, 115200, timeout=1)  # Use the identified serial port

def send_emotion(emotion):
    mqtt_client.publish(MQTT_TOPIC, emotion)
    print(f"Published emotion: {emotion}")

print("Listening for data on serial port...")

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            send_emotion(line)  # Publish only the emotion detected
        mqtt_client.loop()
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    time.sleep(0.1)
