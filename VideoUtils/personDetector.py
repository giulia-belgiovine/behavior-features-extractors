import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Person detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_faces(frame, draw_face=True):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 7)

    if draw_face:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return face_boxes

def detect_person(frame, draw_boxes=True):

    frame = frame[0:750, :]

    person_boxes, weights = hog.detectMultiScale(frame, winStride = (4,4), padding = (4, 4), scale = 1.05)
    person_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in person_boxes])
    pick = non_max_suppression(person_boxes, probs=None, overlapThresh=0.10)
    c = 1

    if draw_boxes:
        for (x, y, w, h) in pick:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (x, y), (w, h), (139, 34, 104), 2)
            cv2.rectangle(frame, (x, y - 20), (w, y), (139, 34, 104), -1)
            cv2.putText(frame, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            c += 1

        cv2.putText(frame, f'Total Persons : {c - 1}', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('output', frame)

    return person_boxes




