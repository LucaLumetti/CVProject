import cv2
import numpy as np
import argparse
from mask_detection import find_mask
from warpface import warp_face

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ask")
    parser.add_argument("--input_front", type=str, help="The input image", required=True)
    parser.add_argument("--input_lateral", type=str, help="The input image", required=True)
    parser.add_argument("--output_dir", type=str, help="Where to save the output images", required=True)

    args = parser.parse_args()
    front = cv2.imread(args.input_front)
    lateral = cv2.imread(args.input_lateral)

    mask = find_mask(front, save_mask=f'{args.output_dir}/mask.jpg')
    warped = warp_face(front, lateral)

    mask = mask[..., np.newaxis]
    # put the warped face over the front one by following the mask
    result = (1-mask)*front + mask*warped

    cv2.imwrite(f'{args.output_dir}/face.jpg', (1-mask)*front)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)

