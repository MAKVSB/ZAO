import sys
import cv2
import numpy as np
import glob
from operator import add
import os
from skimage.feature import local_binary_pattern
from itertools import product
from time import time

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


def four_point_transform(image, one_c):
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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

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


def calculate_sobel(image, kernel_size = 3): 
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def brightness(image, dim=10):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, _, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L)

def main(params):
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
    target_size = (100, 100)

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    print(pkm_coordinates)
    print("********************************************************")

    train = "tryload"
    filename_prefix = str(dict(zip(parameters.keys(), params))).replace(" ", "_").replace("{", "").replace("}", "").replace(":", "")

    if train == "tryload":
        if not os.path.exists(filename_prefix + ".xml"):
            train = "forced"
        else:
            train = "load"

    if train == "forced":
        free_spaces = []
        full_spaces = []

        free_files = os.listdir("data/free/")
        full_files = os.listdir("data/full/")

        for file in free_files:
            image = cv2.imread(f"data/free/{file}", 0)
            free_spaces.append(image)

        for file in full_files:
            image = cv2.imread(f"data/full/{file}", 0)
            full_spaces.append(image)

        spaces_labels = ([0] * len(free_spaces)) + ([1] * len(full_spaces))
        spaces_data = free_spaces + full_spaces

        space_recog = cv2.face.LBPHFaceRecognizer.create(**dict(zip(parameters.keys(), params)))
        space_recog.train(spaces_data, np.array(spaces_labels))
        space_recog.save(filename_prefix + ".xml")
    elif train == "load":
        space_recog = cv2.face.LBPHFaceRecognizer.create()
        space_recog.read(filename_prefix + ".xml")
    else:
        if not os.path.exists(filename_prefix + ".xml"):
            os.makedirs("data")

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow("vokno", 0)
    cv2.namedWindow("vauto", 0)
    cv2.namedWindow("edges", 0)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    time_start = time()

    for img_name in test_images:

        print(img_name)
        image = cv2.imread(img_name, 1)

        correctData = open(img_name.replace('.jpg', '.txt'), 'r').readlines()
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for i, coord in enumerate(pkm_coordinates):
            is_occupied = False
            is_free = False

            car = four_point_transform(original_image, coord)
            car = cv2.resize(car, target_size)

            res = space_recog.predict(car)
            # if (res[1] < 120):
            if (res[0] == 1):
                is_occupied = True
            else:
                is_free = True

            color = (255, 0, 0)
            if (is_occupied):
                color = (0, 0, 255)
            elif (is_free):
                color = (0, 255, 0)
            midpoint1 = midpoint((int(pkm_coordinates[i][0]), int(pkm_coordinates[i][1])), (int(pkm_coordinates[i][4]), int(pkm_coordinates[i][5])))
            midpoint2 = midpoint((int(pkm_coordinates[i][2]), int(pkm_coordinates[i][3])), (int(pkm_coordinates[i][6]), int(pkm_coordinates[i][7])))  
            cv2.putText(image, str(i), midpoint(midpoint1, midpoint2), font, 1, color, 3, cv2.LINE_AA)
            cv2.line(image, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), color, 2)
            cv2.line(image, (int(coord[2]), int(coord[3])), (int(coord[4]), int(coord[5])), color, 2)
            cv2.line(image, (int(coord[4]), int(coord[5])), (int(coord[6]), int(coord[7])), color, 2)
            cv2.line(image, (int(coord[6]), int(coord[7])), (int(coord[0]), int(coord[1])), color, 2)

            cv2.imshow("vokno", image)
            if (int(is_occupied) == int(correctData[i])):
                true_positives += 1
            elif int(is_occupied) == 1 & int(correctData[i]) == 0:
                false_positives += 1
            else :
                false_negatives += 1
        cv2.waitKey(1)
    time_end = time()

    f1 = calculate_f1_score(true_positives, false_positives, false_negatives)
    print("F1 score:", f1, params)
    space_recog.save(filename_prefix + ".xml")
    return (f1, (time_end - time_start)/24)

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
        f1_scores[params] = main(params)

        sorted_f1_scores = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        for params, f1 in sorted_f1_scores:
            print(f"Parameters: {params}, F1 Score: {f1}")




