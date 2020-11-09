
# TODO use opencv to capture camera
import sys
import os
import cv2
from PIL import Image

directory = '/var/tmp/blinkNotifier'

video = cv2.VideoCapture(0)

while True:
    #import time

    #start = time.process_time()
    _, frame = video.read()
    new_height = 128
    dsize = (round(frame.shape[1]/frame.shape[0]*new_height), new_height)
    frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)

    #todo frame is already array with 'bgr', just feed it into the cropping model to separate the face


    cv2.imshow("Capturing", frame)
    # this cv2.waitKey 75ms will work as sleep() in some way
    key = cv2.waitKey(75)
    if key == ord('q'):
        break

    #print(time.process_time() - start)

video.release()
cv2.destroyAllWindows()

# TODO passing the images to a toy keras model

# TODO modify the toy model to a easy classification model for human (find a pretrained one)

# TODO train my own model for classifying blinking and not blinking
