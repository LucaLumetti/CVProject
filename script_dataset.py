#!/usr/bin/python3
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
    cv2.fillPoly(mask, np.array([keypoints]), (255, 255, 255))
    if debug:
        cv2.imshow('mask',mask)
        cv2.waitKey(0)
    return mask

def get_photos(pathname):

    if not os.path.exists(pathname + "masked_images"):
        os.mkdir(os.path.join(pathname, "masked_images"))

    directories = [f.stem for f in Path(pathname).glob("**/*") if f.is_dir() and f.resolve().stem != 'masked_images']

    for dir in directories:
        if not os.path.exists(pathname + "masked_images/" + dir):
            os.mkdir(os.path.join(pathname, "masked_images/" + dir))

    photos = [f for f in Path(pathname).glob("*/*") if f.is_file()]

    return sorted(photos)

if __name__ == '__main__':
    photos = get_photos("dataset/")
    for elem in photos:
        pathname = elem.resolve().as_posix()
        photo_index = elem.stem
        print(pathname)
        print(photo_index)
        img = cv2.imread(pathname)
        cv2.imshow('source', img)
        cv2.waitKey(0)
        mask = create_mask(img, True)
        im = Image.fromarray(mask)
        im.save('dataset/masked_images/' + photo_index + '/' + photo_index + '.png')
        break
