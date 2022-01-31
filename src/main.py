import os
import cv2
import numpy as np
import argparse
from mask_detection import find_mask
from warpface import warp_face

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = f"{current_dir}/../output"

    parser = argparse.ArgumentParser(description="ask")
    parser.add_argument("--input_front", type=str, help="The input image", required=True)
    parser.add_argument("--input_lateral", type=str, help="The reference image")

    args = parser.parse_args()

    front = cv2.imread(args.input_front)
    mask = find_mask(front, save_mask=f'{output_dir}/mask.jpg', debug=True)
    mask = mask[..., np.newaxis]
    cv2.imwrite(f'{output_dir}/face.jpg', (1-mask)*front)
    cv2.imwrite(f'{output_dir}/mask.jpg', mask*255)

    if args.input_lateral is not None:
        lateral = cv2.imread(args.input_lateral)
        warped = warp_face(front, lateral)

        # put the warped face over the front one by following the mask
        result = (1-mask)*front + mask*warped
        cv2.imwrite(f'{output_dir}/face_ref.jpg', (1-mask)*front)

