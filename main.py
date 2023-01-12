class BehaviorAnalyzer:

    def __init__(self, args):
        self.run = False

        # self.humanPose_estimator = HumanPose(args.model_humanpose_path, args.model_hand_path, args.label_hand_path)
        # self.headPose_estimator = HeadPose("/home/icub/PycharmProjects/ROS_humanSensing/src/headpose/checkpoint", 0)
        # self.faceExpression_estimator =

        self.run = True


    def callback(selfself, data):
        if self.run:
            try:
               pass

            except:
                pass



def main():
    rospy.init_node('human_sensing', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("modality", help="Select the modality: DEMO or Video Analysis",
                        type=str)
    parser.add_argument("Face expression model", help = "", type=str, default="/home/icub/PycharmProjects/FaceChannel/TrainedNetworks/CategoricalFaceChannel.h5")
    parser.add_argument("video_path", help="Name of the path of the videos to analyze",
                        type=str)
    parser.add_argument("csv_path", help="Name of the path where to save CSV output",
                        type=str)

    args = parser.parse_args()

    ba = BehaviorAnalyzer(args)
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")
    # cv2.destroyAllWindows()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

