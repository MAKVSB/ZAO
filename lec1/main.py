import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("WebAgg")

from oop.window import Window, WindowFlags
from oop.plotter import Plotter
from oop.image import Image, ImageFormat

print(cv2.__version__)

def windowed():
    window = Window("win-010")
    window2 = Window("win-02")
    window3 = Window("win-03")

    image = Image.from_file("img.png")
    image_gray = Image.from_image(image).convert_format(ImageFormat.GRAY)
    image_gray = image_gray.resize(100,100)

    image_hc = Image.hconcat([image, image_gray])
    image_hc.write('img_hc.png')
    image.show(window)
    image_gray.show(window2)
    image_hc.show(window3)

    cv2.waitKey(0)

def ploted():
    image = Image.from_file("img.png")
    image2 = Image.from_file("img-2.png").convert_format(ImageFormat.GRAY)

    plot = Plotter(2, 1)
    image.show(plot, 0, 0)
    plot.set(image2, 0, 1)
    plot.show()

def cv_2():
    cap = cv2.VideoCapture(0) # int = webcam, string = cesta k videu
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))    

    vw = cv2.VideoWriter("webcam.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))
    
    while True:
        _, frame = cap.read()
        img = Image(frame, ImageFormat.BGR)
        edges = img.canny(50, 100)
        edges.show("edges")
        edges.write(vw)

        if cv2.waitKey(2) == ord('q'):
            break
    vw.release()
    cap.release()

# windowed()
# ploted()
cv_2()