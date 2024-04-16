import cv2
import numpy as np
import time
import os
from matplotlib import pyplot as plt 
import numpy as np  
import matplotlib
matplotlib.use("WebAgg")
import cv2 as cv
from glob import glob
from pathlib import Path
from sklearn.metrics import f1_score

def create_binary_array(length, number_pairs):
    binary_array = np.zeros(length, dtype=int)
    for start, end in number_pairs:
        binary_array[start:end+1] = 1
    return binary_array

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Compute the Euclidean distance between the horizontal eye landmark
    C = np.linalg.norm(eye[0] - eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

display = False

faceDetector = cv2.CascadeClassifier("models/lbpcascade_frontalface_improved.xml")
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("models/opencv_LBF686_GTX.yaml")

for dataset in glob("./datasets/*", recursive = True):
    driverstats = {
        "total": 0,
        "eyes_closed_in_row": 0,
        "danger_zones": []
    }
    faceimages = glob(dataset + "/*.jpg", recursive = True)
    for imagePath in faceimages:
        if display:
            imageData = cv2.imread(imagePath, 1)
            imageDataBW = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        else:
            imageDataBW = cv2.imread(imagePath, 0)

        danger = True

        faces = faceDetector.detectMultiScale(imageDataBW, 1.1, 3, minSize=(80, 80))
        driverstats["total"] += 1
        for face in faces:
            _, landmarks = landmark_detector.fit(imageDataBW, faces)

            for landmark in landmarks:
                if display:
                    for x,y in landmark[0][0:100]:
                        cv2.circle(imageData, (int(x), int(y)), 1, (255, 255, 255), 1)
            
            if display:
                cv2.rectangle(imageData, face,(0, 0, 255), 2)

            if eye_aspect_ratio(landmark[0][36:42]) > 0.2 and eye_aspect_ratio(landmark[0][42:48]) > 0.2:
                danger = False

        if danger:
            driverstats["eyes_closed_in_row"] += 1
        else:
            if (driverstats["eyes_closed_in_row"] > 20):
                driverstats["danger_zones"].append((driverstats["total"] - driverstats["eyes_closed_in_row"], driverstats["total"]))
            driverstats["eyes_closed_in_row"] = 0

        if display:
            cv2.imshow("test", imageData)
            cv2.waitKey(1)



    correct_results = [x.replace("\n", "").split(" ") for x in open(dataset + '/anot.txt', 'r').readlines()]
    correct_results = [(int(x[0]), int(x[1])) for x in correct_results]

    originalArray = create_binary_array(driverstats["total"], correct_results)
    detectedArray = create_binary_array(driverstats["total"], driverstats["danger_zones"])

    print(correct_results)
    print(driverstats["danger_zones"])

    f1 = f1_score(originalArray, detectedArray)

    print("F1 Score:", f1)