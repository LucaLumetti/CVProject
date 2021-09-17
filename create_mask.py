import cv2
import numpy as np
import matplotlib.pyplot as plt
import thinplate as tps
import random
import mediapipe as mp
from skimage import draw
from os import listdir
from os.path import isfile, join

mypath = "./photos/"

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

# Return image with the same size of the original image but with 0 out
# of the mask
def get_mask(img, mask):
    for i in range(mask.shape[0]):
        mask[i][0] *= img.shape[1]
        mask[i][1] *= img.shape[0]

    mask = np.around(mask)
    mask = mask.astype(int)

    hullIndex = cv2.convexHull(mask, returnPoints=False)

    lnd = []

    for c in range(0, len(hullIndex)):
        lnd.append(mask[int(hullIndex[c])])

    cropped_img = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillConvexPoly(cropped_img, np.int32(lnd), (1.0, 1.0, 1.0), 16, 0)
    return cropped_img*img

if __name__ == "__main__":
    photos = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for photo in photos:
        img = cv2.imread(photo)

        # init mediapipe net for facemesh
        mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        key_point = faceMesh.process(img)
        landmarks = key_point.multi_face_landmarks[0]

        landmarks = np.array([
            [landmark.x, landmark.y]
            for landmark in landmarks.landmark
        ])

        landmarks = landmarks[bottom_face_landmarks]

        img_mask = get_mask(img, landmarks)
