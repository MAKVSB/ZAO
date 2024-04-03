import cv2
import numpy as np
import time

def cv_2():
    cap = cv2.VideoCapture(0) # int = webcam, string = cesta k videu
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))    

    vw = cv2.VideoWriter("webcam.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))
    vw.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vw.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cc_frontal = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    cc_profile = cv2.CascadeClassifier("haarcascades/haarcascade_profileface.xml")
    cc_eye = cv2.CascadeClassifier("haarcascades/eye_cascade_fusek.xml")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        display_image = frame.copy()

        faces = cc_profile.detectMultiScale(frame, 1.1, 5)
        for face in faces:
            cv2.rectangle(display_image, face, (0, 255, 0), 2)

        faces = cc_frontal.detectMultiScale(frame, 1.1, 5)
        for face in faces:
            cv2.rectangle(display_image, face, (0, 255, 0), 2)
            frame_face_part = frame[face[1]:face[1]+face[2], face[0]:face[0]+face[3]]
            display_frame_face_part = frame_face_part.copy()
            eye = cc_eye.detectMultiScale(frame_face_part, 1.1, 5)[0:2]
            for e in eye:
                eye_part = display_frame_face_part[e[1]:e[1]+e[2], e[0]:e[0]+e[3]]
                if (e[0]+e[2] < face[2]/2):
                    color = (0, 0, 255)
                    cv2.imshow("l_eye",eye_part)
                else:
                    color = (255, 0, 0)
                    cv2.imshow("r_eye",eye_part)

                eye_part_gray = cv2.cvtColor(eye_part, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(eye_part_gray, cv2.HOUGH_GRADIENT, 1, eye_part.shape[0] / 15, param1=150, param2=20)
                print(circles)
                if (circles is not None):
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        cv2.circle(eye_part, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    color = (color[0], 255, color[2])

                cv2.rectangle(display_frame_face_part, e, color, 2)
                e[0] = e[0] + face[0]
                e[1] = e[1] + face[1]
                cv2.rectangle(display_image, e, color, 2)

            cv2.imshow("face", display_frame_face_part)
        cv2.imshow("frame", display_image)
        end_time = time.time()

        print(end_time- start_time)

        if cv2.waitKey(2) == ord('q'):
            break
    vw.release()
    cap.release()

cv_2()