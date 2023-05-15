#source code a: https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from tqdm import tqdm

from HeadPose.faceUtils import faceProcessing

faceProcessing = faceProcessing.faceProcessing()
mp_drawing = mp.solutions.drawing_utils

class HeadPose:

    def __init__(self, method):
        self.method = method
        self.head_pose_detectors = {
            "mediapipe": self.headpose_mediapipe, # TODO usare un enum
            "open_face": self.headpose_openface,
        }

    def set_method(self, method):
        self.method = method

    def run(self, frame):
        return self.head_pose_detectors[self.method](frame)

    def headpose_openface(self, frame):
        return frame, _, _, _, _

    def headpose_mediapipe(self, frame):
        #To improve performance
        frame.flags.writeable = False
        # Convert the color space from RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the result
        #results_face = faceProcessing.faceDetector.process(frame)     #if you want to use the mediapipe facedetector
        results = faceProcessing.face_mesh.process(frame)           #if you want to use the mediapipe mesh face

        # Draw the face detection annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = frame.shape

        faceProcessing.compute_camera_parameters(img_w, img_h)

        # img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []
        text = ""
        x, y, z = -999, -999, -999

        # if results_face.detections:
        #     print('ok')

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D and 3D Coordinates
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Compute camera matrix
                faceProcessing.compute_camera_parameters(img_w, img_h)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, faceProcessing.camera_matrix, faceProcessing.dist_coeffs)
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting
                text = faceProcessing.create_labels(x, y)

                # Display the nose direction
                faceProcessing.display_output(frame, text, nose_2d, x, y, z)
                # faceProcessing.draw_mesh(frame, face_landmarks)

        return frame, text, x, y, z






