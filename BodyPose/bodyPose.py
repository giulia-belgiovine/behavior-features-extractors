import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from tqdm import tqdm


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

class BodyPose:

    def __init__(self, method):
        self.method = method
        self.body_pose_detectors = {
            "mediapipe": self.bodypose_holistic
            #"open_pose": self.open_pose,
        }

        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5)


    def set_method(self, method):
        self.method = method

    def run(self, frame, debug_mode):
        return self.body_pose_detectors[self.method](frame, debug_mode)

    def bodypose_holistic(self, image):
        # To improve performance
        image.flags.writeable = False
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

            # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

        return image, results.pose_landmarks





