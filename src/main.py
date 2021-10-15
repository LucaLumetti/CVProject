import cv2
import numpy as np
from mask_detection import find_mask
from warpface import warp_face

if __name__ == '__main__':
    front = cv2.imread('front_lateral_people/1_front_m.jpg')
    lateral = cv2.imread('front_lateral_people/1_lat.jpg')

    mask = find_mask(front)
    warped = warp_face(front, lateral)

    mask = mask[..., np.newaxis]
    # put the warped face over the front one by following the mask
    result = (1-mask)*front + mask*warped
    cv2.imshow('result', result)
    cv2.waitKey(0)

