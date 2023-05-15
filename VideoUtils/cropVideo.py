import numpy as np
import cv2
import os

# Open the video
video_path = "/home/icub/Desktop/output/"
video_path_cropped = "/home/icub/Desktop/prova"
video_name = [f for f in os.listdir(video_path) if f.endswith(".avi")][0]
video = os.path.join(video_path, video_name)
cropped_video = os.path.join(video_path_cropped, video_name.split('.')[0] + "_cropped.avi")

cap = cv2.VideoCapture(video)

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Here you can define your cropping values
x,y,h,w = 10,25,455,350  #1100,250,400,500  #
finalImageSize = (w, h) #(1100, 780)

# output
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(cropped_video, fourcc, 30, finalImageSize)

# Now we start
while(cap.isOpened()):
    ret, frame = cap.read()
    cnt += 1
    # Avoid problems when video finish
    if ret:
        crop_frame = frame[y:y+h, x:x+w]
        print(crop_frame.shape)

        out.write(crop_frame)
        cv2.imshow('cropped',crop_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()