#!/usr/bin/python3
import sys
import cv2
import numpy as np
import os
import csv
from mask_detection import find_facial_landmarks, get_image
from pathlib import Path
from PIL import Image

def create_mask(img, debug=False):
    keypoints = find_facial_landmarks(img)
    mask = np.zeros(img.shape[:2], np.uint8)
    if len(keypoints) == 0:
        return None
    cv2.fillPoly(mask, np.array([keypoints]), (255, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    mask =  cv2.dilate(mask, kernel, iterations=10)
    if debug:
        cv2.imshow('mask',mask)
        cv2.waitKey(0)
    return mask

def get_photos(pathname):
    if not os.path.exists(pathname + "masked_images"):
        os.mkdir(os.path.join(pathname, "masked_images"))

    directories = [f.stem for f in Path(pathname).glob("FFHQ/**/*") if f.is_dir() and f.resolve().stem != 'masked_images']

    for dir in directories:
        if not os.path.exists(pathname + "masked_images/" + dir):
            os.mkdir(os.path.join(pathname, "masked_images/" + dir))

    photos = [f for f in Path(pathname).glob("FFHQ/*/*") if f.is_file()]

    return sorted(photos)

if __name__ == '__main__':
    pathname = Path(sys.argv[1])
    split_folder = sys.argv[1].split('/')
    subfolder = split_folder[-2]
    filename = split_folder[-1]
    path_to_mask = Path(f'{pathname.parent.parent.parent}' + \
            f'/masked_images/{subfolder}/{filename}')
    path_to_mask.parent.mkdir(parents=True, exist_ok=True)

    if path_to_mask.exists():
        exit(0)

    img = cv2.imread(str(pathname))

    mask = create_mask(img, False)
    if mask is not None:
        cv2.imwrite(str(path_to_mask), mask)
    else:
        print(f"Face not found for {pathname}")
