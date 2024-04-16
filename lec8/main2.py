import cv2
import dlib
import faceBlendCommon as face
import numpy as np
# Load Image
im = cv2.imread("face.jpg")
# Detect face landmarks
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor("./../lec7/models/dlib_shape_predictor_68_face_landmarks.dat")
landmarks = face.getLandmarks(faceDetector, landmarkDetector, im)
# Create a mask for the lips
lipsPoints = landmarks[0:27]
print(lipsPoints)
mask = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)
cv2.fillConvexPoly(mask, np.int32(lipsPoints), (1.0, 1.0, 1.0))
mask = 255*np.uint8(mask)
# Apply close operation to improve mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
cv2.imshow("test", mask)
# Blur the mask to obtain natural result
mask = cv2.GaussianBlur(mask,(15,15),cv2.BORDER_DEFAULT)
# Calculate inverse mask
inverseMask = cv2.bitwise_not(mask)
# Convert masks to float to perform blending
mask = mask.astype(float)/255
inverseMask = inverseMask.astype(float)/255
# Apply color mapping for the lips
lips = cv2.applyColorMap(im, cv2.COLORMAP_INFERNO)
# Convert lips and face to 0-1 range
lips = lips.astype(float)/255
ladyFace = im.astype(float)/255
# Multiply lips and face by the masks
justLips = cv2.multiply(mask, lips)
justFace = cv2.multiply(inverseMask, ladyFace)
# Add face and lips
result = justFace + justLips
# Show result
cv2.imshow("", result)
cv2.waitKey(0)