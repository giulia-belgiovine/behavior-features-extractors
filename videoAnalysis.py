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


class BehavioralPipeline:
    def __init__(self, r, v):

        self.root_dir = r
        self.video_format = v
        self.source_dir = os.path.join(self.root_dir, 'input_videos')  # Folder where the videos to process are
        self.out_dir = os.path.join(self.root_dir, 'behavioral_analysis_output')  # Folder that will hold the output folders

        self.head_output_path = os.path.join(self.out_dir, 'head_pose')
        self.emotions_output_path = os.path.join(self.out_dir, 'emotions')
        self.body_output_path = os.path.join(self.out_dir, 'body_pose')
        self.flow_output_path = os.path.join(self.out_dir, 'optical_flow')

        self.head_analysis = False
        self.emotion_analysis = False
        self.body_analysis = False
        self.optical_flow_analysis = False

        self.head_pose_extractor = None
        self.emotion_extractor = None
        self.body_pose_extractor = None
        self.optical_flow_extractor = None

        self.detect_single_individual = True

        self.width = 0
        self.height = 0
        self.fps = 30

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
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = int(cap.get(cv2.CAP_PROP_FPS))
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height
                final_image_size = (self.width, self.height)
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

                print("Started Video:" + str(video) + " - Total Frames:" + str(total_frames))

                ########## HEAD ##########
                if self.head_analysis:
                    if not os.path.exists(self.head_output_path):
                        os.makedirs(self.head_output_path)

                ########## BODY ##########
                if self.body_analysis:
                    if not os.path.exists(self.body_output_path):
                        os.makedirs(self.body_output_path)

                ########## OPTICAL FLOW ##########
                if self.optical_flow_analysis:
                    if not os.path.exists(self.flow_output_path):
                        os.makedirs(self.flow_output_path)


                with open(self.out_dir + "/" + video.split('.')[0] + ".csv", mode='w') as employee_file:
                    writer_employee = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer_employee.writerow(["Frame", "Head_Angles", "Face_Heading_Label", "Joints_Position", "Optical_Flow_Magnitude"])

                    # Save output video
                    result_video_head = cv2.VideoWriter(self.head_output_path + "/" + video[:-4] + '.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                                                        self.fps, final_image_size)
                    result_video_body = cv2.VideoWriter(self.body_output_path + "/" + video[:-4] + '.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                                                        self.fps, final_image_size)
                    result_video_flow = cv2.VideoWriter(self.flow_output_path + "/" + video[:-4] + '.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                                                        self.fps, final_image_size)


                    ########## PROCESS VIDEOS ###########
                    with tqdm(total=total_frames) as pbar:
                        while (cap.isOpened() and not frame_count == total_frames):

                            pbar.set_description("Process video")
                            ret, frame = cap.read()
                            frame_count = frame_count + 1
                            if type(frame) is np.ndarray:

                                # Detect person in the frame
                                if self.detect_single_individual:
                                    person_boxes = detect_person(frame, draw_boxes=True)
                                    for (xA, yA, xB, yB) in person_boxes:
                                        person_frame = frame[yA:yB, xA:xB]


                                if self.optical_flow_analysis and self.optical_flow_extractor is not None:
                                    image_flow, magnitude, prev_gray = self.optical_flow_extractor.run(frame, prev_gray, mask)
                                    sum_magnitude_frame = np.sum(magnitude[0:750,:])
                                    # cv2.imshow('Final Image Flow', image_flow)


                                if self.head_analysis and self.head_pose_extractor is not None:
                                    image_head, head_label, x, y, z = self.head_pose_extractor.run(frame)
                                    cv2.putText(image_head, f'FPS: {int(frame_count)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                                 1.5, (255, 255, 0), 2)
                                    # display output frame for DEBUG
                                    cv2.imshow('Final Image Head', image_head)


                                if self.emotion_analysis and self.emotion_extractor is not None:
                                    ret_val = self.emotion_extractor.run(frame)


                                if self.body_analysis and self.body_pose_extractor is not None:
                                    image_body, joints = self.body_pose_extractor.run(frame)
                                    joints_to_save = joints.landmark[11:24]
                                    # display output frame for DEBUG
                                    cv2.imshow('Final Image Body', image_body)


                                writer_employee.writerow([int(frame_count), [x, y, z], head_label, joints_to_save, sum_magnitude_frame])
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break

                                # save videos
                                result_video_head.write(image_head)
                                result_video_body.write(image_body)
                                result_video_flow.write(image_flow)

                            pbar.update(1)

                        video_time = (time.time() - video_time)

            print("Finished Video: " + str(video) + " - Time:" + str(video_time) + " seconds")
            result_video_flow.release()
            cap.release()
            cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root_dir", help="Name of the path where input_videos folder is", default='home/icub/Desktop/gioca_jouer', type=str)
    parser.add_argument("-v", "--video_format", help="Format of the videos to be processed", default='.mp4', type=str)
    args = parser.parse_args()

    bp = BehavioralPipeline(args.root_dir, args.video_format)
    bp.extract_head_pose("mediapipe")
    bp.extract_optical_flow("dense_franeback")
    bp.extract_emotion("default")
    bp.extract_body_pose("mediapipe")

    bp.ingest_videos()


if __name__ == '__main__':
    main()
