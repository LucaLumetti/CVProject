import cv2
import numpy as np
import matplotlib.pyplot as plt
import thinplate as tps
import random
import mediapipe as mp
from skimage import draw
from os import listdir, walk
from os.path import isfile, join

import csv

csv_file="./photo_name.csv"

mypath = "./dataset/"

bottom_face_landmarks = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48, 49, 50, 51, 57, 58,
        59, 60, 61, 62, 64, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102,
        106, 115, 123, 126, 129, 131, 132, 134, 135, 136, 137, 138, 140, 141,
        142, 146, 147, 148, 149, 150, 152, 164, 165, 166, 167, 169, 170, 171,
        172, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
        191, 192, 194, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
        209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 235, 236,
        237, 238, 239, 240, 241, 242, 248, 250, 262, 266, 267, 268, 269, 270,
        271, 272, 273, 274, 278, 279, 288, 289, 290, 291, 292, 294, 302, 303,
        304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
        318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 331, 335, 344,
        352, 355, 358, 360, 361, 363, 364, 365, 366, 367, 369, 371, 375, 376,
        377, 378, 379, 391, 392, 393, 394, 395, 396, 397, 400, 401, 402, 403,
        404, 405, 406, 407, 408, 409, 410, 411, 415, 416, 418, 420, 421, 422,
        423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436,
        437, 438, 439, 440, 455, 456, 457, 458, 459, 460, 462]


fixed_landmarks = [234, 454, 447, 345, 346, 347, 348, 349, 277, 437, 399, 419,
        197, 196, 174, 47, 120, 119, 118, 117, 116, 34, 127]

def find_facial_landmarks(img, landmarks=[], debug=False):

    with open('landmarks_list.txt', 'r') as f:
        landmarks_list = [int(i) for i in f.readline().strip().split(',')]

    keypoints = []

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    img_land = img.copy()
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img_land,
                    faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            #for id,lm in enumerate(faceLms.landmark):
            #    print(lm)
            #    ih, iw, ic = img.shape
            #    x,y = int(lm.x*iw), int(lm.y*ih)
            #    print(id, x,y)
        ih,iw,ic = img.shape

        for landmark in landmarks_list:
            xc = int(faceLms.landmark[landmark].x*iw)
            yc = int(faceLms.landmark[landmark].y*ih)

            if not landmark in fixed_landmarks:
                yc += 40
            else:
                yc -= 10
            keypoints.append((xc, yc))
    if debug:
        cv2.imshow('landmarks', img_land)
        cv2.waitKey(0)
    return keypoints

def find_mask(img, debug=True  ):

    # Keypoints detection
    keypoints = find_facial_landmarks(img, debug=debug)
    if not keypoints:
        return np.array([])
    mask = np.zeros(img.shape, dtype=img.dtype)

    cv2.fillConvexPoly(mask, np.int32(keypoints), (255, 255, 255), 16, 0)

    return mask

if __name__ == "__main__":
    # Get the list of all files in directory tree at given path
    photos = list()
    for (dirpath, dirnames, filenames) in walk(mypath):
        photos += [join(dirpath, file) for file in filenames]

    with open(csv_file, "w") as csv_name:
        writer = csv.writer(csv_name)
        writer.writerow(photos)

    for photo in photos:
        img = cv2.imread(photo)

        '''
        cv2.imshow("original", img)
        cv2.waitKey(0)
        '''

        img_mask = find_mask(img)

        mask_name = photo.split(".", 2)
        mask_name = "."+mask_name[1] + "_mask."+mask_name[2]

        cv2.imshow("mask", img_mask)
        cv2.waitKey(0)
        cv2.imwrite(mask_name, img_mask)
