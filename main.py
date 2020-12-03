# TODO use opencv to capture camera
import cv2
import numpy as np

directory = '/var/tmp/blinkNotifier'

video = cv2.VideoCapture(0)

# OPENCV_DNN
modelFile = "model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

import dlib


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


predictor = dlib.shape_predictor('model/shape_68.dat')

while True:
    # import time

    # start = time.process_time()
    _, frame = video.read()
    new_height = 360
    dsize = (round(frame.shape[1] / frame.shape[0] * new_height), new_height)
    frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    max_index = 0
    if faces[0, 0, max_index, 2] > 0.7:
        box = faces[0, 0, max_index, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

    # todo frame is already array with 'bgr', just feed it into the cropping model to separate the face

    # todo: try dlib and keypoint
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(left=x, top=y, right=x1, bottom=y1)
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Capturing", frame)
    # this cv2.waitKey 75ms will work as sleep() in some way
    key = cv2.waitKey(75)
    if key == ord('q'):
        break



    # print(time.process_time() - start)

video.release()
cv2.destroyAllWindows()

# TODO passing the images to a toy keras model

# TODO modify the toy model to a easy classification model for human (find a pretrained one)

# TODO train my own model for classifying blinking and not blinking
