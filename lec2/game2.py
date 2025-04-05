import cv2
import matplotlib
matplotlib.use("WebAgg")
from PIL import ImageGrab  
import pyautogui
import time

from oop.window import Window
from oop.image import Image, ImageFormat

from selenium import webdriver

# Set the path to the Vivaldi executable
game_url = 'https://duckhuntjs.com'
repeat_offset = 90
up_constant = 0.2

options = webdriver.ChromeOptions()
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--window-size=800,800")
driver = webdriver.Chrome(options=options)
driver.set_window_position(0, 0, windowHandle='current')
driver.get(game_url)

cv_window = Window("win1").resize(800, 800).move(800,0)
cv_window_zoomed = Window("win1").resize(800, 800).move(800,800)

def click_to_pos(x: int, y: int):
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(0.2)

def analyze_zoomed(center_position_x:int, center_position_y:int, templateImage):
    rep_print_screen = Image.from_object(ImageGrab.grab(bbox=(center_position_x-repeat_offset, center_position_y-repeat_offset, center_position_x+repeat_offset, center_position_y+repeat_offset)), ImageFormat.BGR).convert_format(ImageFormat.GRAY)
    rep_match_data = rep_print_screen.matchTemplate(templateImage, cv2.TM_CCOEFF_NORMED)
    (_, inner_max_val, _, inner_max_loc) = cv2.minMaxLoc(rep_match_data.img)
    return (inner_max_val, inner_max_loc, rep_print_screen)

def get_zoomed_diff(position:tuple[int, int], templateImage):
    return (
        int(position[0] + templateImage.width()/2 - repeat_offset),
        int(position[1] + templateImage.height()/2 - repeat_offset)
    )

windowPos = driver.get_window_position()
windowPos = (windowPos["x"], windowPos["y"])

windowSize = driver.get_window_size()
windowSize = (windowSize["width"], windowSize["height"])

cv_window = cv_window.resize(int(windowSize[0] / 2), windowSize[1]).move(windowSize[0],0)

templates = []
for i in  range(0, 11):
    image = Image.from_file(f"ducks/{i}.png", color=True)
    templates.append(image.convert_format(ImageFormat.GRAY))

while True: # mainLoop
    windowPos = driver.get_window_position()
    windowPos = (windowPos["x"], windowPos["y"])

    windowSize2 = driver.get_window_size()
    windowSize2 = (windowSize2["width"], windowSize2["height"])
    if windowSize2 != windowSize:
        print("rescaling")
        xScale = windowSize2[0] / windowSize[0]
        yScale = windowSize2[1] / windowSize[1]
        print(xScale, yScale)
        windowSize = windowSize2
        cv_window.move(windowSize[0],0)
        cv_window_zoomed.move(windowSize[0],windowSize[1])
        for template in templates:
            template.resize(int(template.width()*xScale), int(template.height()*yScale))
        repeat_offset = templates[0].width()

    for i in range(0, len(templates)):
        for j in range(0, 1):
            printScreen = Image.from_object(ImageGrab.grab(bbox=(windowPos[0], windowPos[0] + windowSize[1] * up_constant, windowPos[0] + windowSize[0], windowPos[0] + windowSize[1] *0.9)), ImageFormat.BGR).convert_format(ImageFormat.GRAY)
            match_data = printScreen.matchTemplate(templates[i], cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(match_data.img)
            if max_val > 0.7:
                printScreen.convert_format(ImageFormat.RGB)
                centerPosX = max_loc[0] + int(templates[i].width()/2)
                centerPosY = max_loc[1] + int(templates[i].height()/2) + int( windowSize[1] * up_constant)

                subset = 3* int(i/3)

                for k in range(0, 3):
                    print(i, subset, subset+k)
                    (inner_max_val, inner_max_loc, rep_print_screen) = analyze_zoomed(centerPosX, centerPosY, templates[subset+k])
                    (diffX, diffY) = get_zoomed_diff(inner_max_loc, templates[subset+k])
                    if inner_max_val > 0.7:
                        click_to_pos(centerPosX + diffX, centerPosY + diffY )
                        cv2.circle(printScreen.img, (centerPosX + diffX, centerPosY + diffY), 10, (0, 255, 0), 2)
                        rep_print_screen.img = cv2.circle(rep_print_screen.img, inner_max_loc, 10, (0, 0, 255), 3)
                        rep_print_screen.show(cv_window_zoomed)
                        cv2.waitKey(2)
                        break
                    rep_print_screen.img = cv2.circle(rep_print_screen.img, (centerPosX + diffX, centerPosY + diffY), 10, (0, 0, 255), 3)
                    rep_print_screen.show(cv_window_zoomed)
                    cv2.waitKey(2)
            printScreen.show(cv_window)
            cv2.waitKey(2)