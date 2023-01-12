import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import GREEN_COLOR, WHITE_COLOR
from mediapipe.python.solutions.drawing_styles import _THICKNESS_DOT
from HeadPose.FaceUtils import faceProcessing

# For webcam input:
cap = cv2.VideoCapture(0)
faceProcessing = faceProcessing.faceProcessing()
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    success, image = cap.read()

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceProcessing.faceDetector.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape

    faceProcessing.compute_camera_parameters(img_w, img_h)

    if results.detections:

        features = np.ndarray((len(results.detections), 11), dtype=np.float64)

        for idx, detection in enumerate(results.detections):
            features[idx, :] = faceProcessing.compute_2_5D_from_landmarks_(detection, img_w, img_h)

            faceProcessing.mp_drawing.draw_detection(image, detection)
            c_arrow = faceProcessing.draw_head_direction_line(image, detection, features[idx, :], img_w, img_h)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
