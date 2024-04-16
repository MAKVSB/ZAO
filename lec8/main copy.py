import cv2
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
matplotlib.use("WebAgg")
from glob import glob

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Compute the Euclidean distance between the horizontal eye landmark
    C = np.linalg.norm(eye[0] - eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

faceDetector = cv2.CascadeClassifier("./../lec7/models/lbpcascade_frontalface_improved.xml")

landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("./../lec7/models/opencv_LBF686_GTX.yaml")

swapFace = cv2.imread("face.jpg", 1)
swapFacesBW = cv2.cvtColor(swapFace, cv2.COLOR_BGR2GRAY)
swapFaceData = faceDetector.detectMultiScale(swapFacesBW, 1.1, 3, minSize=(80, 80))[0]
swapFaceCopy = swapFace.copy()
swapFace = swapFace[swapFaceData[1]:swapFaceData[1] + swapFaceData[3], swapFaceData[0]:swapFaceData[0] + swapFaceData[2]]

for dataset in glob("./../lec7/datasets/*", recursive = True):
    for imagePath in glob(dataset + "/*.jpg", recursive = True):
        imageData = cv2.imread(imagePath, 1)
        imageDataBW = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(imageDataBW, 1.1, 3, minSize=(80, 80))
        for face in faces:
            resized_image = cv2.resize(swapFace, (face[2], face[3]))
            output1 = imageData.copy()
            output1[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = resized_image
            cv2.imshow("output1", output1)

            output2 = imageData.copy()
            src_face_mask = np.zeros(resized_image.shape,dtype=resized_image.dtype)
            src_face_mask.fill(255)
            center_face = (face[0] + face[2] // 2, face[1] + face[3] // 2)
            output2 = cv2.seamlessClone(resized_image, output2, src_face_mask, center_face, cv2.MONOCHROME_TRANSFER)
            cv2.imshow("output2", output2)

            _, landmarks = landmark_detector.fit(swapFacesBW, np.array([(0, 0, swapFace.shape[0], swapFace.shape[1])]))
            landmarks = np.int32(landmarks)[0][0]
            landmarks = [(x[0], x[1]) for x in landmarks[0:27]]














            im = swapFace.copy()
            mask = np.zeros((im.shape[0]+30, im.shape[1]+30, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(landmarks), (1.0, 1.0, 1.0))
            mask = 255*np.uint8(mask)
            # Apply close operation to improve mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
            mask = mask[0:im.shape[0], 0:im.shape[1]]

            mask = cv2.resize(mask, (face[2], face[3]))


            cv2.imshow("test", mask)







            print(src_face_mask.shape)
            print(mask.shape)


            output3 = imageData.copy()
            center_face = (face[0] + face[2] // 2, face[1] + face[3] // 2)
            output3 = cv2.seamlessClone(resized_image, output3, mask, center_face, cv2.MONOCHROME_TRANSFER)










            cv2.imshow("output3", output3)
        cv2.waitKey(1)

