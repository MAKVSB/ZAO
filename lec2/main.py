import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("WebAgg")
from PIL import ImageGrab  
import pyautogui
import time

from oop.window import Window, WindowFlags
from oop.plotter import Plotter
from oop.image import Image, ImageFormat

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin

# Set the path to the Vivaldi executable
game_url = 'https://www.addictinggames.com/shooting/dart-master'
template_path = "template.png"
driver_path = "C:/School/6Sem/ZAO/chromedriver.exe"
vivaldi_path = "C:/School/6Sem/ZAO/chrome-win64/chrome.exe"
repeat_offset = 80

service = Service(executable_path=driver_path)
options = webdriver.ChromeOptions()
options.binary_location = vivaldi_path
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--window-size=1000,800")
driver = webdriver.Chrome(service=service, options=options)
driver.set_window_position(0, 0, windowHandle='current')
driver.get(game_url)
# driver.get('file:///C:/School/6Sem/ZAO/lec2/main.py')

templateImage = Image.from_file(template_path).convert_format(ImageFormat.GRAY)
cv_window = Window("win1").resize(1000, 800).move(1000,0)

def click_to_pos(x: int, y: int):
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(0.6)

def analyze_zoomed(center_position_x:int, center_position_y:int):
    rep_print_screen = Image.from_object(ImageGrab.grab(bbox=(center_position_x-repeat_offset, center_position_y-repeat_offset, center_position_x+repeat_offset, center_position_y+repeat_offset)), ImageFormat.BGR).convert_format(ImageFormat.GRAY)
    rep_match_data = rep_print_screen.matchTemplate(templateImage, cv2.TM_CCOEFF_NORMED)
    (_, inner_max_val, _, inner_max_loc) = cv2.minMaxLoc(rep_match_data.img)
    return (inner_max_val, inner_max_loc)

def get_zoomed_diff(position:tuple[int, int]):
    return (
        int(position[0] + templateImage.width()/2 - repeat_offset),
        int(position[1] + templateImage.height()/2 - repeat_offset)
    )

def predict_position(center_position_x:int, center_position_y: int, multiplier:int = 3):
    (_, inner_max_loc) = analyze_zoomed(center_position_x, center_position_y)
    diff_from_center = get_zoomed_diff(inner_max_loc)
    center_position_x += diff_from_center[0]
    center_position_y += diff_from_center[1]
    predicted_position_x = center_position_x + diff_from_center[0]*multiplier
    predicted_position_y = center_position_y + diff_from_center[1]*multiplier
    return (predicted_position_x, predicted_position_y)


windowPos = driver.get_window_position()
windowPos = (windowPos["x"], windowPos["y"])

windowSize = driver.get_window_size()
windowSize = (windowSize["width"], windowSize["height"])

cv_window = cv_window.resize(int(windowSize[0] / 2), windowSize[1]).move(windowSize[0],0)

while True: # mainLoop
        windowPos = driver.get_window_position()
        windowPos = (windowPos["x"], windowPos["y"])

        windowSize = driver.get_window_size()
        windowSize = (windowSize["width"], windowSize["height"])

        printScreen = Image.from_object(ImageGrab.grab(bbox=(windowPos[0], windowPos[0], windowPos[0] + windowSize[0], windowPos[0] + windowSize[1])), ImageFormat.BGR).convert_format(ImageFormat.GRAY)
        matchData = printScreen.matchTemplate(templateImage, cv2.TM_CCOEFF_NORMED)
        printScreen.convert_format(ImageFormat.BGR).show(cv_window)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchData.img)
        centerPositionX = int(max_loc[0] + templateImage.width()/2 + windowPos[0])
        centerPositionY = int(max_loc[1] + templateImage.height()/2 + windowPos[0])

        print(max_val)
        while max_val > 0.90: # just because i try to follow the same one if previous attempt fails
            printScreen = Image.from_object(ImageGrab.grab(bbox=(windowPos[0], windowPos[0], windowPos[0] + windowSize[0], windowPos[0] + windowSize[1])), ImageFormat.BGR).convert_format(ImageFormat.GRAY)
            printScreen.convert_format(ImageFormat.BGR).show(cv_window)
            printScreen.img = cv2.circle(printScreen.img, (centerPositionX, centerPositionY), 10, (0, 0, 255), 3)
            printScreen.show(cv_window)
            cv2.waitKey(2)
            # animation detection
            animating = False
            missed = 0
            for i in range(0,3): # static target
                (inner_max_val, inner_max_loc) = analyze_zoomed(centerPositionX, centerPositionY)
                diffFromCenter = get_zoomed_diff(inner_max_loc)
                centerPositionX += diffFromCenter[0]
                centerPositionY += diffFromCenter[1]
                printScreen.img = cv2.circle(printScreen.img, (centerPositionX, centerPositionY), 10, (0, 255, 0), 3)
                printScreen.show(cv_window)
                cv2.waitKey(2)
                print("diff: ", diffFromCenter, "missed:", missed)

                if sum([abs(ele) for ele in diffFromCenter]) > 1:
                    missed += 1
            if missed > 2:
                animating = True

            if not animating:
                click_to_pos(centerPositionX, centerPositionY)
                break
            else:
                diffs = []

                for i in range(0,5): # prediction with revalidation x times
                    (predictedPositionX, predictedPositionY) = predict_position(centerPositionX, centerPositionY)
                    printScreen.img = cv2.circle(printScreen.img, (centerPositionX, centerPositionY), 5, (255, 0, 0), 3)
                    printScreen.show(cv_window)
                    cv2.waitKey(2)
                    time.sleep(0.15)
                    (inner_max_val, inner_max_loc) = analyze_zoomed(centerPositionX, centerPositionY)
                    diffFromCenter = get_zoomed_diff(inner_max_loc)
                    centerPositionX += diffFromCenter[0]
                    centerPositionY += diffFromCenter[1]
                    diff = abs(centerPositionX - predictedPositionX) + abs(centerPositionY - predictedPositionY)
                    printScreen.img = cv2.circle(printScreen.img, (centerPositionX, centerPositionY), 5, (255, 255, 0), 3)
                    printScreen.show(cv_window)
                    cv2.waitKey(2)
                    diffs.append(diff)

                diff = sum(diffs)/len(diffs)                
                if diff < 5:
                    centerPositionX += diffFromCenter[0]*3
                    centerPositionY += diffFromCenter[1]*3
                    click_to_pos(centerPositionX, centerPositionY)
                    break

                print("trying speed") # record min max speed and shoot the speed in middle of them with 2x delta
                # try find min speed and shoot then
                minspeed = 100
                maxspeed = 0
                for i in range(0,30):
                    (inner_max_val, inner_max_loc) = analyze_zoomed(centerPositionX, centerPositionY)

                    diffFromCenter = get_zoomed_diff(inner_max_loc)
                    centerPositionX += diffFromCenter[0]
                    centerPositionY += diffFromCenter[1]

                    speed = round(np.sqrt(diffFromCenter[0]**2 + diffFromCenter[1]**2))
                    if speed < minspeed:
                        minspeed = speed
                    if speed > maxspeed:
                        maxspeed = speed
                print("minspeed: ", minspeed)
                print("maxspeed: ", maxspeed)

                for i in range(0,100):
                    (inner_max_val, inner_max_loc) = analyze_zoomed(centerPositionX, centerPositionY)
                    diffFromCenter = get_zoomed_diff(inner_max_loc)
                    centerPositionX += diffFromCenter[0]
                    centerPositionY += diffFromCenter[1]

                    speed = round(np.sqrt(diffFromCenter[0]**2 + diffFromCenter[1]**2))
                    if speed == maxspeed-minspeed:
                        centerPositionX += diffFromCenter[0]
                        centerPositionY += diffFromCenter[1]
                        click_to_pos(centerPositionX, centerPositionY)
                        break
                else:
                    continue
                break
                
                # youre fucked






        printScreen.show(cv_window)


        if cv2.waitKey(2) == ord('q'):
            break

driver.quit()
cv2.destroyAllWindows()