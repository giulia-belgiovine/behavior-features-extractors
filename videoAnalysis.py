import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from tqdm import tqdm
from HeadPose.headPose import HeadPose

from mediapipe.python.solutions.drawing_utils import GREEN_COLOR, WHITE_COLOR
from mediapipe.python.solutions.drawing_styles import _THICKNESS_DOT

loadVideosFrom = "/home/icub/Desktop/social_exclusion/cropped_videos" #Folder where the videos are
saveCSVFiles = "/home/icub/Desktop/social_exclusion/output_head" #Folder that will hold the .csv files
width= 500 #640
height=400  #480
fps = 30



class BehavioralPipeline:
    def __init__(self):
        self.head_pose_extractor = None
        self.emotion_extractor = None
        self.body_pose_extractor = None

    def extract_head_pose(self, method):
        self.head_pose_extractor = HeadPose(method)

    def extract_emotion(self, method):
        pass

    def extract_body_pose(self, method):
        pass

    def ingest_videos(self, width, height, fps, source_dir, out_dir):
        finalImageSize = (int(width), int(height))
        for videoDirectory in os.listdir(source_dir):

            if videoDirectory.endswith('.avi'):

                videoTime = time.time()
                # Open the video
                cap = cv2.VideoCapture(loadVideosFrom + "/" + videoDirectory)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Opens the .csv file #####
                with open(saveCSVFiles + "/" + videoDirectory.split('.')[0] + ".csv", mode='w') as employee_file:
                    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    employee_writer.writerow(["Frame", "Player", "Angles", "Face_heading_label"])
                    frameCount = 0

                    player_label = ["Green" if "green" in str(videoDirectory) else "Blue"]

                    print("Started Video:" + str(videoDirectory) + " - Total Frames:" + str(total))
                    # Save output video
                    result_video = cv2.VideoWriter(saveCSVFiles + "/" + videoDirectory[:-4] + '.avi',
                                                   cv2.VideoWriter_fourcc(*'MJPG'), fps, finalImageSize)

                    with tqdm(total=total) as pbar:
                        while (cap.isOpened() and not frameCount == total):

                            pbar.set_description("Process video")
                            ret, frame = cap.read()
                            frameCount = frameCount + 1
                            if type(frame) is np.ndarray:

                                if self.head_pose_extractor is not None:
                                    image, head_label, x, y, z = self.head_pose_extractor.run(frame)
                                    employee_writer.writerow([int(frameCount), player_label, [x, y, z], head_label])
                                    cv2.putText(image, f'FPS: {int(frameCount)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                                1.5, (255, 255, 0), 2)

                                if self.emotion_extractor is not None:
                                    ret_val = self.head_pose_extractor.run(frame)   # TODO

                                if self.body_pose_extractor is not None:
                                    ret_val = self.body_pose_extractor.run(frame)   # TODO

                                # E poi boh, quello che devi fare


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



def main():
    pipeline = BehavioralPipeline()
    pipeline.extract_head_pose("mediapipe")
    pipeline.extract_emotion("default")
    pipeline.extract_body_pose("default")
    pipeline.ingest_videos(width, height, fps, source_dir=loadVideosFrom, out_dir=saveCSVFiles)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("modality", help="Select the modality: DEMO or Video Analysis",
    #                     type=str)
    # parser.add_argument("Face expression model", help = "", type=str, default="/home/icub/PycharmProjects/FaceChannel/TrainedNetworks/CategoricalFaceChannel.h5")
    # parser.add_argument("video_path", help="Name of the path of the videos to analyze",
    #                     type=str)
    # parser.add_argument("csv_path", help="Name of the path where to save CSV output",
    #                     type=str)
    #
    # args = parser.parse_args()
    #
    # ba = BehaviorAnalyzer(args)


if __name__ == '__main__':
    main()
