# Obj_Detect
Overview
This project implements an obstacle detection system using a webcam feed and the YOLOv5 object detection model. The system identifies specific obstacles and uses text-to-speech to alert the user about the presence of these obstacles in real-time.

Requirements
Python 3.x
OpenCV
PyTorch
pyttsx3
Pathlib
Installation
Clone the Repository:

sh
git clone https://github.com/yourusername/obstacle-detection.git
cd obstacle-detection
Install Dependencies:

sh
pip install opencv-python torch pyttsx3 pathlib
Download COCO Names:
Ensure you have the coco.names file in the same directory as your script. You can download it from COCO dataset labels.

Usage
Run the Script:

sh
python obstacle_detection.py
Interact with the System:

The script will use your webcam by default. If you want to use a video file, replace cap = cv2.VideoCapture(0) with the path to your video file, e.g., cap = cv2.VideoCapture('video.mp4').
The system detects obstacles such as person, chair, table, rock, fire hydrant, and sofa.
If an obstacle is detected, it will be outlined with a bounding box and the system will announce the obstacle using text-to-speech.
Press the 'q' key to exit the program.
Code Explanation
Initialization
python
Copy code
import cv2
import torch
from pathlib import Path
import pyttsx3

engine = pyttsx3.init()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
coco_names = Path('coco.names').read_text().strip().split('\n')
obstacle_classes = ['person', 'chair', 'table', 'rock', 'fire hydrant', 'sofa']
cap = cv2.VideoCapture(0)
clear_path = True
obstacle_detected = False
Import necessary libraries.
Initialize text-to-speech engine.
Load the YOLOv5 model.
Read COCO class names.
Define obstacle classes of interest.
Initialize video capture and state variables.
Main Loop
python
Copy code
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    clear_path = True

    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, xyxy)
            label = coco_names[int(cls)]
            if label in obstacle_classes:
                color = (0, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if not obstacle_detected:
                    engine.say(f"{label} ahead")
                    engine.runAndWait()
                    obstacle_detected = True
                clear_path = False

    if clear_path:
        obstacle_detected = False

    cv2.imshow('Obstacle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Capture frames from the webcam.
Use the YOLOv5 model to detect objects in the frame.
If an obstacle is detected, draw a bounding box and announce it.
Display the frame with detections.
Break the loop on pressing 'q'.
Notes
Ensure the coco.names file is present in the same directory as the script.
Adjust the confidence threshold and obstacle classes as needed.
Make sure your webcam is connected and accessible.
