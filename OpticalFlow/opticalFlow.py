import cv2
import numpy as np
import os
import time
import csv
from tqdm import tqdm


class OpticalFlow:

    def __init__(self, method):
        self.method = method
        self.optical_flow_detectors = {
            "dense_franeback": self.optical_flow_dense
        }


    def set_method(self, method):
        self.method = method

    def run(self, frame, prev_gray, mask):
        return self.optical_flow_detectors[self.method](frame, prev_gray, mask, debug_mode)

    def optical_flow_dense(self, frame, prev_gray, mask, debug_mode):
        # Converts frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        sum_magnitude_frame = np.sum(magnitude[0:750, :])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        image_flow = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        return image_flow, sum_magnitude_frame, gray