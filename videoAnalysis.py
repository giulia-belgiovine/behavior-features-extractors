import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from tqdm import tqdm

from mediapipe.python.solutions.drawing_utils import GREEN_COLOR, WHITE_COLOR
from mediapipe.python.solutions.drawing_styles import _THICKNESS_DOT
from HeadPose.faceUtils import faceProcessing
from HeadPose import headPose

loadVideosFrom = "/home/icub/Desktop/social_exclusion/cropped_videos" #Folder where the videos are
saveCSVFiles = "/home/icub/Desktop/social_exclusion/output_head" #Folder that will hold the .csv files


faceProcessing = faceProcessing.faceProcessing()
headPose = headPose.HeadPose(flag=True)

width= 500 #640
height=400  #480
fps = 30
finalImageSize = (int(width), int(height))


for videoDirectory in os.listdir(loadVideosFrom):

    if videoDirectory.endswith('.avi'):

        videoTime = time.time()
        #Open the video
        cap = cv2.VideoCapture(loadVideosFrom + "/" + videoDirectory)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #Opens the .csv file #####
        with open(saveCSVFiles + "/" + videoDirectory.split('.')[0] + ".csv", mode='w') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow(["Frame", "Player", "Angles", "Face_heading_label"])
            frameCount = 0

            player_label = ["Green" if "green" in str(videoDirectory) else "Blue"]

            print("Started Video:" + str(videoDirectory) + " - Total Frames:" + str(total))
            #Save output video
            result_video = cv2.VideoWriter(saveCSVFiles + "/" + videoDirectory[:-4] + '.avi',
                                           cv2.VideoWriter_fourcc(*'MJPG'), fps, finalImageSize)

            with tqdm(total=total) as pbar:
                while(cap.isOpened() and not frameCount == total):

                    pbar.set_description("Process video")
                    ret, frame = cap.read()
                    frameCount = frameCount + 1
                    if type(frame) is np.ndarray:

                        image, head_label, x, y, z = headPose.run(frame)
                        employee_writer.writerow([int(frameCount), player_label, [x, y, z], head_label])
                        cv2.putText(image, f'FPS: {int(frameCount)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

                        result_video.write(image)
                        cv2.imshow('Head Pose Estimation', image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    pbar.update(1)

        videoTime = (time.time() - videoTime)

        print("Finished Video: " + str(videoDirectory) + " - Time:" + str(videoTime) + " seconds")
        result_video.release()
        cap.release()
        cv2.destroyAllWindows()


