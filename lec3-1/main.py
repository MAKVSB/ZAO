import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob
from operator import add

from oop.window import Window, WindowFlags
from oop.plotter import Plotter
from oop.image import Image, ImageFormat

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image: Image, one_c)-> Image:
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = Image(cv2.warpPerspective(image.img, M, (maxWidth, maxHeight)))
    # return the warped image
    return warped

def midpoint(p1, p2):
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

def calculate_f1_score(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    if precision + recall == 0:
        return 0 
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


def main(argv):
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
    target_size = (80, 80)

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    print(pkm_coordinates)
    print("********************************************************")

    templates = []
    templates_global = []
    templates_global_paths = glob.glob("templates/global_*.jpg")

    for tmpl in templates_global_paths:
        tmplt = cv2.imread(tmpl, 0)
        tmplt = cv2.resize(tmplt, target_size)
        templates_global.append(tmplt)


    for i, coord in enumerate(pkm_coordinates):
        tmplts = glob.glob( "templates/" + str(i) + '_*.jpg')
        tmplts.sort()
        templates.append([])
        for tmpl in tmplts:
            tmplt = cv2.imread(tmpl, 0)
            tmplt = cv2.resize(tmplt, target_size)
            templates[i].append(tmplt)

    font = cv2.FONT_HERSHEY_SIMPLEX
    vokno = Window("vokno", 0)
    vauto = Window("vauto", 0)

    for img_name in test_images:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        print(img_name)
        image = Image.from_file(img_name, True)
        correctData = open(img_name.replace('.jpg', '.txt'), 'r').readlines()
        original_image = image.copy().convert_format(ImageFormat.GRAY)

        for i, coord in enumerate(pkm_coordinates):
            car = four_point_transform(original_image, coord)
            car.img = cv2.resize(car.img, target_size)

            is_occupied = True
            for j, tmpl in enumerate(templates[i] + templates_global):
                res = cv2.matchTemplate(car.img, tmpl, cv2.TM_SQDIFF_NORMED)
                if (cv2.minMaxLoc(res)[0] < 0.065):
                    is_occupied = False

            midpoint1 = midpoint((int(pkm_coordinates[i][0]), int(pkm_coordinates[i][1])), (int(pkm_coordinates[i][4]), int(pkm_coordinates[i][5])))
            midpoint2 = midpoint((int(pkm_coordinates[i][2]), int(pkm_coordinates[i][3])), (int(pkm_coordinates[i][6]), int(pkm_coordinates[i][7])))   




            color = (0, 255, 0)
            if (is_occupied):
                color = (0, 0, 255)
            
            cv2.putText(image.img, str(i), [x + y for x, y in zip(midpoint(midpoint1, midpoint2), (-20, 0))], font, 1, color, 3, cv2.LINE_AA)
            cv2.line(image.img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), color, 2)
            cv2.line(image.img, (int(coord[2]), int(coord[3])), (int(coord[4]), int(coord[5])), color, 2)
            cv2.line(image.img, (int(coord[4]), int(coord[5])), (int(coord[6]), int(coord[7])), color, 2)
            cv2.line(image.img, (int(coord[6]), int(coord[7])), (int(coord[0]), int(coord[1])), color, 2)



            image.show(vokno)
            car.show(vauto)
            if (int(is_occupied) == int(correctData[i])):
                true_positives += 1
            elif int(is_occupied) == 1 & int(correctData[i]) == 0:
                false_positives += 1
                cv2.waitKey(0)
            else :
                false_negatives += 1
                cv2.waitKey(0)

        print("F1 score:", calculate_f1_score(true_positives, false_positives, false_negatives))
        cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv[1:])
