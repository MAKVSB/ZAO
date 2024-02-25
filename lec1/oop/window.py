import cv2
from enum import Enum
import time


class WindowFlags(Enum):
    WINDOW_NORMAL = cv2.WINDOW_NORMAL, 
    WINDOW_AUTOSIZE = cv2.WINDOW_AUTOSIZE, 
    WINDOW_OPENGL = cv2.WINDOW_OPENGL, 
    WINDOW_FULLSCREEN = cv2.WINDOW_FULLSCREEN, 
    WINDOW_FREERATIO = cv2.WINDOW_FREERATIO, 
    WINDOW_KEEPRATIO = cv2.WINDOW_KEEPRATIO, 
    WINDOW_GUI_EXPANDED = cv2.WINDOW_GUI_EXPANDED, 
    WINDOW_GUI_NORMAL = cv2.WINDOW_GUI_NORMAL 

class Window:
    def __init__(self, name: str, flags: int = 0):
        self.name = name + str(time.time())
        self.win = cv2.namedWindow(self.name, flags)
    
    def resize(self, width: int, height: int)-> 'Window':
        cv2.resizeWindow(self.name, width, height)
        return self
