import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
import argparse
from tqdm import tqdm
from HeadPose.headPose import HeadPose
from BodyPose.bodyPose import BodyPose
from OpticalFlow.opticalFlow import OpticalFlow
from VideoUtils.personDetector import detect_person

class VideoWriterManager:
    def __init__(self, out_dir):
        self.streams = {}
        self.out_dir = out_dir

    def add_stream(self, stream_name):
        self.streams[stream_name] = None

    def init(self, name, codec, fps, out_size):
        for k in self.streams.keys():
            self.streams[k] = cv2.VideoWriter(f"{self.out_dir}/{k}/{name}.avi", codec, fps, out_size)

    def write(self, stream_name, frame):
        if stream_name in self.streams:
            self.streams[stream_name].write(frame)
            cv2.imshow("output_frame", frame)

    def reset(self):
        for k in self.streams.keys():
            self.streams[k] = None




class BehavioralPipeline:
    def __init__(self, root, video_format, head, emotion, body, flow, debug):

        self.root_dir = root
        self.video_format = video_format

        # The videos to analyze should be inside "input_videos" folder inside the root_dir
        self.source_dir = os.path.join(self.root_dir, 'input_videos')
        self.out_dir = os.path.join(self.root_dir, 'behavioral_analysis_output')

        self.video_writer = VideoWriterManager(self.out_dir)

        self.head_analysis = head
        self.emotion_analysis = emotion
        self.body_analysis = body
        self.optical_flow_analysis = flow

        self.head_pose_extractor = None
        self.emotion_extractor = None
        self.body_pose_extractor = None
        self.optical_flow_extractor = None
        self.detect_single_individual = True

        self.debug = debug


        if self.debug and self.head_analysis:
            self.video_writer.add_stream("head")
        if self.debug and self.emotion_analysis:
            self.video_writer.add_stream("emotion")
        if self.debug and self.body_analysis:
            self.video_writer.add_stream("body")
        if self.debug and self.optical_flow_analysis:
            self.video_writer.add_stream("optical_flow")

        self.fps = 30
        self.final_image_size = (0, 0)
        self.total_frames = 0
        self.video_name = ""
        self.video_time = 0


    def extract_video_features(self, cap, video):
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.final_image_size = (width, height)
        self.video_name = video[:-4]


    def extract_head_pose(self, method):
        self.head_pose_extractor = HeadPose(method)

    def extract_emotion(self, method):
        pass

    def extract_body_pose(self, method):
        self.body_pose_extractor = BodyPose(method)

    def extract_optical_flow(self, method):
        self.optical_flow_extractor = OpticalFlow(method)


    def ingest_videos(self):

        for video in os.listdir(self.source_dir):

            if video.endswith(self.video_format):
                self.video_time = time.time()

                ########## OPEN VIDEO ##########
                cap = cv2.VideoCapture(self.source_dir + "/" + video)
                self.extract_video_features(cap, video)
                frame_count = 0

                #For optical flow
                ret, first_frame = cap.read()
                prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(first_frame)
                mask[..., 1] = 255

                # Initialize variable to save in CSV output
                [x, y, z] = [0,0,0]
                head_label = None
                joints_to_save = None
                sum_magnitude_frame = None

                print("Started Video:" + str(video) + " - Total Frames:" + str(self.total_frames))

                with open(self.out_dir + "/" + video.split('.')[0] + ".csv", mode='w') as employee_file:
                    writer_employee = csv.writer(employee_file, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    writer_employee.writerow(["Frame", "Head_Angles", "Face_Heading_Label", "Joints_Position", "Optical_Flow_Magnitude"]) # Save output video

                    self.video_writer.init(self.video_name, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, self.final_image_size)


                    ########## PROCESS VIDEOS ###########
                    with tqdm(total=self.total_frames) as pbar:
                        while cap.isOpened() and not frame_count == self.total_frames:

                            pbar.set_description("Process video")
                            ret, frame = cap.read()
                            frame_count = frame_count + 1
                            if type(frame) is np.ndarray:

                                # Detect person in the frame
                                # if self.detect_single_individual:
                                #     person_boxes = detect_person(frame, draw_boxes=True)
                                #     for (xA, yA, xB, yB) in person_boxes:
                                #         person_frame = frame[yA:yB, xA:xB]

                                if self.optical_flow_analysis and self.optical_flow_extractor is not None:
                                    out_frame, magnitude, prev_gray = self.optical_flow_extractor.run(frame, prev_gray, mask)
                                    self.video_writer.write("optical_flow", out_frame)

                                if self.head_analysis and self.head_pose_extractor is not None:
                                    out_frame, head_label, x, y, z = self.head_pose_extractor.run(frame)
                                    self.video_writer.write("head", out_frame)

                                if self.emotion_analysis and self.emotion_extractor is not None:
                                    self.emotion_extractor.run(frame)


                                if self.body_analysis and self.body_pose_extractor is not None:
                                    out_frame, image_body, joints = self.body_pose_extractor.run(frame)
                                    joints_to_save = joints.landmark[11:24]
                                    self.video_writer.write(out_frame)

                                writer_employee.writerow([int(frame_count), [x, y, z], head_label, joints_to_save, sum_magnitude_frame])
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break

                            pbar.update(1)

                        self.video_time = (time.time() - video_time)

            print("Finished Video: " + str(video) + " - Time:" + str(self.video_time) + " seconds")
            # cap.release()
            cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser()

    #Specify root path name and video format
    parser.add_argument("-root", "--root_dir", help="Name of the path where input_videos folder is", default='/home/icub/Desktop/gioca_jouer', type=str)
    parser.add_argument("-format", "--video_format", help="Format of the videos to be processed", default='.mp4')

    #Activate analysis you want to perform
    parser.add_argument("-o", "--head", help="Activate head analysis", action="store_true")
    parser.add_argument("-e", "--emotion", help="Activate emotion analysis", action="store_true")
    parser.add_argument("-b", "--body", help="Activate body analysis", action="store_true")
    parser.add_argument("-f", "--flow", help="Activate Optical flow analysis", action="store_true")

    parser.add_argument("-d", "--debug", help="show output and save video", action="store_true")

    args = parser.parse_args()
    bp = BehavioralPipeline(args.root_dir, args.video_format,
                            args.head,
                            args.emotion,
                            args.body,
                            args.flow,
                            args.debug)

    #Specify method for each analysis
    bp.extract_head_pose("mediapipe")
    bp.extract_optical_flow("dense_franeback")
    bp.extract_emotion("default")
    bp.extract_body_pose("mediapipe")

    #Start analysis
    bp.ingest_videos()


if __name__ == '__main__':
    main()
