import cv2 as cv
import numpy as np

img1 = cv.imread("", 0)
img2 = cv.imread("", 0)

img1 = cv.resize(img1, (80,80))
img2 = cv.resize(img2, (80,80))

train_list = [img1, img2]
train_labels = [0,1]

face_recog = cv.face.LBPHFaceRecognizer.create()
print("GridX", face_recog.getGridX())
print("GridY", face_recog.getGridY())
print("Neighbors", face_recog.getNeighbors())
print("Radius", face_recog.getRadius())

face_recog.train(train_list, np.array(train_labels))

img3 = cv.imread("", 0)
img3 = cv.resize(img3, (80,80))
result = face_recog.predict(img3)
print("reslut", result)

