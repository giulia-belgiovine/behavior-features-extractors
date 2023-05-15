import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import GREEN_COLOR, WHITE_COLOR
from mediapipe.python.solutions.drawing_styles import _THICKNESS_DOT


class faceProcessing:

  def __init__(self):

    self.faceDetector = mp.solutions.face_detection.FaceDetection(model_selection="full",
                                                                  min_detection_confidence=0.01)
    self.mp_face_mesh = mp.solutions.face_mesh
    self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5,
                                                max_num_faces=4)

    self.mp_drawing = mp.solutions.drawing_utils
    self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Constant Standard Face Parameters
    self.camera_matrix = None
    self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)


  def draw_mesh(self, frame, face_landmarks):

    self.mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=self.drawing_spec,
        connection_drawing_spec=self.drawing_spec)


  def compute_camera_parameters(self, img_w, img_h):
    self.camera_matrix = np.array([[1 * img_w, 0, img_h / 2],
                           [0, 1 * img_w, img_w / 2],
                           [0, 0, 1]])

  def create_labels(self, x, y):

    text, text_x, text_y = "", "", ""
    if y < -10:
      text_y = "Left"
    elif y > 10:
      text_y = "Right"
    else:
      text = "Forward"

    if x < -10:
      text_x = "Down"
    elif x > 10:
      text_x = "Up"
    else:
      text = "Forward"

    text = text + " " + text_x + " " + text_y

    return text

  def display_output(self, frame, text, nose_2d, x, y, z):

    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    cv2.line(frame, p1, p2, (255, 0, 0), 3)

    # Add the text on the image
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(frame, "x: " + str(np.round(x, 2)), (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "y: " + str(np.round(y, 2)), (500, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "z: " + str(np.round(z, 2)), (500, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



