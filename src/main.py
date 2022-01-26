import cv2
import numpy as np
from mask_detection import find_mask
from warpface import warp_face

if __name__ == '__main__':
    front = cv2.imread('./front_lateral_people/tay_front.jpg')
    lateral = cv2.imread('./front_lateral_people/tay_lat.jpg')

    mask = find_mask(front)
    warped = warp_face(front, lateral)

    mask = mask[..., np.newaxis]

    cv2.imshow('result', (1-mask)*front)
    cv2.imwrite('hole.png', (1-mask)*front)
    cv2.waitKey(0)
    # calculate new mask
    # put the warped face over the front one by following the mask
    result = (1-mask)*front + mask*warped

    mask = mask

    cv2.imshow('result', mask)
    cv2.imwrite('result_mask.png', mask)
    cv2.waitKey(0)

    warped[warped > 0] = 255
    warped[warped == 0] = 1
    warped[warped == 255] = 0

    mask = warped*mask
    result = result + mask*255

    cv2.imshow('result', result)
    cv2.imwrite('result.png', result)
    cv2.waitKey(0)

    cv2.imshow('result', mask*255)
    cv2.imwrite('result_mask.png', mask*255)
    cv2.waitKey(0)

