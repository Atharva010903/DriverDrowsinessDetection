import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the trained model
model = load_model('my_model.h5')

lbl = ['Close', 'Open']

# Ask the user for input type
input_type = input("Enter 'live' for webcam or provide the video file path: ")

if input_type.lower() == 'live':
    cap = cv2.VideoCapture(0)  # Use webcam
else:
    cap = cv2.VideoCapture(input_type)  # Use pre-recorded video

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends
    
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    for (x, y, w, h) in eyes:
        eye = frame[y:y+h, x:x+w]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye)
        print(prediction)
        
        if prediction[0][0] > 0.30:  # Closed eye condition
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            score += 1
            if score > 15:
                try:
                    sound.play()
                except:
                    pass
        elif prediction[0][1] > 0.70:  # Open eye condition
            score -= 1
            if score < 0:
                score = 0
            cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
USING MTCNN:
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from pygame import mixer

# Initialize sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load MTCNN detector
detector = MTCNN()

# Load trained model
model = load_model('my_model.h5')

# Class labels
lbl = ['Closed', 'Open']

# Ask the user for input type
input_type = input("Enter 'live' for webcam or provide the video file path: ")

if input_type.lower() == 'live':
    cap = cv2.VideoCapture(0)  # Use webcam
else:
    cap = cv2.VideoCapture(input_type)  # Use pre-recorded video

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with MTCNN
    faces = detector.detect_faces(frame)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Extract eyes using facial landmarks
        keypoints = face['keypoints']
        eyes = [keypoints['left_eye'], keypoints['right_eye']]

        for ex, ey in eyes:
            eye_x1, eye_y1 = ex - 15, ey - 15
            eye_x2, eye_y2 = ex + 15, ey + 15
            eye = frame[eye_y1:eye_y2, eye_x1:eye_x2]

            if eye.shape[0] > 0 and eye.shape[1] > 0:
                eye = cv2.resize(eye, (80, 80))
                eye = eye / 255.0
                eye = eye.reshape(80, 80, 3)
                eye = np.expand_dims(eye, axis=0)

                prediction = model.predict(eye)
                print(prediction)

                if prediction[0][0] > 0.30:  # Closed eye condition
                    cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    score += 1
                    if score > 15:
                        try:
                            sound.play()
                        except:
                            pass
                elif prediction[0][1] > 0.70:  # Open eye condition
                    score -= 1
                    if score < 0:
                        score = 0
                    cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
