import cv2
import numpy as np
import time
import os

from matplotlib import pyplot as plt 
import numpy as np  
import matplotlib
matplotlib.use("WebAgg")
import cv2 as cv

def cv_2(params):
    eyes_open = []
    eyes_closed = []

    free_files = os.listdir("open/")
    full_files = os.listdir("close/")

    for file in free_files:
        image = cv2.imread(f"open/{file}", 0)
        eyes_open.append(image)

    for file in full_files:
        image = cv2.imread(f"close/{file}", 0)
        eyes_closed.append(image)

    spaces_labels = ([0] * len(eyes_open)) + ([1] * len(eyes_closed))
    eyes_data = eyes_open + eyes_closed

    eye_recog = cv2.face.LBPHFaceRecognizer.create(**dict(zip(parameters.keys(), params)))
    eye_recog.train(eyes_data, np.array(spaces_labels))












    cap = cv2.VideoCapture("fusek_face_car_01.avi") # int = webcam, string = cesta k videu

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))    

    vw = cv2.VideoWriter("webcam.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))
    vw.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vw.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cc_frontal = cv2.CascadeClassifier("../../lec5/haarcascades/haarcascade_frontalface_default.xml")
    cc_profile = cv2.CascadeClassifier("../../lec5/haarcascades/haarcascade_profileface.xml")
    cc_eye = cv2.CascadeClassifier("../../lec5/haarcascades/eye_cascade_fusek.xml")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if ret == False:
            break
        display_image = frame.copy()

        faces = cc_profile.detectMultiScale(frame, 1.1, 5)
        for face in faces:
            cv2.rectangle(display_image, face, (0, 255, 0), 2)

        faces = cc_frontal.detectMultiScale(frame, 1.1, 4, minSize=(400, 400))
        for face in faces:
            cv2.rectangle(display_image, face, (0, 255, 0), 2)
            frame_face_part = frame[face[1]:face[1]+int(face[2]*0.7), face[0]:face[0]+face[3]]
            display_frame_face_part = frame[face[1]:face[1]+face[2], face[0]:face[0]+face[3]]
            eye = cc_eye.detectMultiScale(frame_face_part, 2.3, 4)[0:100]
            for e in eye:
                color = (255, 0, 0)
                eye_part = display_frame_face_part[e[1]+60:e[1]+e[2]-30, e[0]:e[0]+e[3]]

                color = (0, 0, 255)

                if len(eye_part) != 0:
                    result, guess = eye_recog.predict(cv2.cvtColor(eye_part, cv2.COLOR_BGR2GRAY))

                    if result == 0:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                cv2.rectangle(display_frame_face_part, e, color, 2)
                e[0] = e[0] + face[0]
                e[1] = e[1] + face[1]
                cv2.rectangle(display_image, e, color, 2)
        cv2.imshow("frame", display_image)
        end_time = time.time()

        print(end_time- start_time)

        if cv2.waitKey(2) == ord('q'):
            break
    vw.release()
    cap.release()

if __name__ == "__main__":
    parameters = {
        'radius': [1, 2, 3, 4, 5, 6, 7, 8],
        'neighbors': [1, 2, 3, 4, 6, 8, 10, 12, 14], 
        'grid_x': [4, 6, 8, 10, 12],
        'grid_y': [4, 6, 8, 10, 12]
    }

    # parameter_combinations = product(*parameters.values())
    parameter_combinations = [
        (4, 2, 4, 4),
        (3, 8, 4, 4),
        (3, 10, 4, 4)
    ]
    f1_scores = {}

    for params in parameter_combinations:
        print(params)
        f1_scores[params] = cv_2(params)

        sorted_f1_scores = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        for params, f1 in sorted_f1_scores:
            print(f"Parameters: {params}, F1 Score: {f1}")