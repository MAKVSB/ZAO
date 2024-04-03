import cv2
import matplotlib
matplotlib.use("WebAgg")

from oop.window import Window, WindowFlags
from oop.plotter import Plotter
from oop.image import Image, ImageFormat

cap = cv2.VideoCapture("kor-4-1.avi")
window1 = Window("debug")
window5 = Window("contours")

while True:
    ret, frame = cap.read()
    frame_gray = Image(frame, ImageFormat.BGR).convert_format(ImageFormat.GRAY)
    frame_tresh = frame_gray.copy().threshold(127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    frame_dil = frame_tresh.dilate(kernel)
    frame_er = frame_dil.erode(kernel)
    contours = frame_er.findContours(cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    omg = frame_gray.copy().convert_format(ImageFormat.RGB).drawContours(contours, -1, (0, 255, 0), 3)

    result = Image.show_grid([frame_gray, frame_tresh, frame_dil, frame_er, omg], "max", "grid")


    result.resize(1024, 640).show(window1)
    omg.resize(1024, 640).show(window5)

    if cv2.waitKey(2) == ord('q'):
        break
