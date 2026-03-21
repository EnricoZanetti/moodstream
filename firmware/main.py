"""
This script is designed to run on the OpenMV Cam H7 Plus. It captures images using the camera,
detects faces, classifies the emotions of the detected faces using a pre-trained TensorFlow Lite model,
and then sends the detected emotions over UART. The emotions are transmitted as strings, which can be 
read by a connected device such as a PC.

Key functionalities:
1. Initialize the camera and set it to capture images in QVGA format.
2. Load a pre-trained TensorFlow Lite model for emotion detection.
3. Capture images and detect faces.
4. Classify the emotions of the detected faces.
5. Send the detected emotions via UART.
"""

import sensor, image, time, tf, pyb

# Initialize UART
uart = pyb.UART(3, 115200, timeout_char=1000)
print("UART initialized")

def send_emotion(label):
    print(f"Emotion: {label}")
    uart.write(label + "\n")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 2000)
print("Sensor initialized")

clock = time.clock()
print("Loading model...")
net = tf.load("trained.tflite", load_to_fb=True)
labels = [l.rstrip('\n') for l in open("labels.txt")]
print("Model loaded and labels read")

while(True):
    clock.tick()

    # Take a picture and brighten things up for the frontal face detector
    img = sensor.snapshot().gamma_corr(contrast=1.5)

    # Returns a list of rects (x, y, w, h) where faces are
    faces = img.find_features(image.HaarCascade("frontalface"))

    for f in faces:
        # Classify a face and get the class scores list
        scores = net.classify(img, roi=f)[0].output()

        # Find the highest class score and lookup the label for that
        label = labels[scores.index(max(scores))]

        # Draw a box around the face
        img.draw_rectangle(f)

        # Draw the label above the face
        img.draw_string(f[0]+3, f[1]-1, label, mono_space=False)

        # Send the detected emotion via UART
        send_emotion(label)
