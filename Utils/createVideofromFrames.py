import numpy
import cv2
import os
import glob
from io import BytesIO
import numpy as np


loadFramesFrom = "/home/icub/Desktop/social_exclusion/cropped_videos"
saveVideosin = "/home/icub/Desktop/social_exclusion/output_emotion"
size=(640,480)


for folder in os.listdir(loadFramesFrom):
    out = cv2.VideoWriter(saveVideosin + folder + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    img_array = []
    file_log = open((os.path.join(loadFramesFrom, folder, 'data.log')), 'r')
    lines = file_log.read().splitlines()

    for line in lines:
        filename = os.path.join(loadFramesFrom, folder, line.split(' ')[2])
        image = cv2.imread(filename)
        cv2.imshow('', image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        height, width, layers = image.shape
        size = (width, height)
        img_array.append(image)
        del(image)


    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



out.release()