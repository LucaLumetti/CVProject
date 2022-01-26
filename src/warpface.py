import cv2
import numpy as np
import matplotlib.pyplot as plt
import thinplate as tps
import random
import mediapipe as mp
from skimage import draw

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

landmarks_for_tps = [
        # from left ear to right ear
        93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
        365, 397, 288, 361, 323, 137, 123, 50, 134, 51, 5,
        # mid line of the face
        # 0,1,2,3,4,5,19,24,164,11,12,13,14,15,16,17,18,200,199,175,
        # mouth
        61,91,
        # cheecks
        123,280,
    ]

left_face = [
        152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 0, 1, 17, 423, 280,
        459, 290,
        ]

# Apply the thin plate spline to the image
def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

# Return image with the same size of the original image but with 0 out
# of the mask
def crop_image(img, mask):
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

def trasformation_image(dst, cropped_img, lnd_src, lnd_dst):
    # TODO: this MUST be improved, landmarks are normalized/denormalized so many
    # time, we can avoid this
    for i in range(lnd_src.shape[0]):
        # lnd_src[i][0] *= cropped_img.shape[1]
        # lnd_src[i][1] *= cropped_img.shape[0]
        lnd_dst[i][0] *= dst.shape[1]
        lnd_dst[i][1] *= dst.shape[0]

    norm_lnd_src = np.copy(lnd_src)
    norm_lnd_dst = np.copy(lnd_dst)

    norm_lnd_src[:,0] /= cropped_img.shape[1]
    norm_lnd_src[:,1] /= cropped_img.shape[0]

    norm_lnd_dst[:,0] /= dst.shape[1]
    norm_lnd_dst[:,1] /= dst.shape[0]

    src_adapting = warp_image_cv(cropped_img, norm_lnd_src, norm_lnd_dst)
    # cv2.waitKey(0)

    # lnd_dst = np.around(lnd_dst)
    # lnd_dst = lnd_dst.astype(int)

    # hullIndex = cv2.convexHull(np.array(lnd_dst), returnPoints=False)

    # lnd = []

    # for c in range(0, len(hullIndex)):
    #     lnd.append(lnd_dst[int(hullIndex[c])])

    # base = np.copy(dst).astype(np.uint8)
    # cv2.fillConvexPoly(base, np.int32(lnd), (0, 0, 0), 16, 0)

    # output = base+src_adapting

    # # Clone seamlessly.
    # mask = np.zeros(dst.shape, dtype=dst.dtype)
    # cv2.fillConvexPoly(mask, np.int32(lnd), (255, 255, 255), 16, 0)
    # r = cv2.boundingRect(np.float32([lnd]))
    # center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    # output = cv2.seamlessClone(np.uint8(src_adapting), base, mask, center, cv2.NORMAL_CLONE)
    # M = np.float32([
    #     [1, 0, -26],
    #     [0, 1, 0]
    # ])
    # src_adapting = src_adapting + cv2.warpAffine(cv2.flip(src_adapting, 1), M, (src_adapting.shape[1], src_adapting.shape[0]))
    # cv2.imshow('src_adapting', src_adapting)
    # cv2.waitKey(0)
    return src_adapting

# would maybe be better if we take from lateral as much as we can to cover the
# mask
def warp_face(front, lateral, debug=False):
    # make the shape the same, this maybe can be removed in the future
    lateral = cv2.resize(lateral, (front.shape[1], front.shape[0]))

    # init mediapipe net for facemesh
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    frontalFaceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    drawSpec = mpDraw.DrawingSpec(color=(255,0,0),thickness=1, circle_radius=1)
    lateralFaceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

    # process
    front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
    # lateral = cv2.cvtColor(lateral, cv2.COLOR_BGR2RGB)
    front_kp = frontalFaceMesh.process(front)
    lateral_kp = lateralFaceMesh.process(lateral)

    print(front_kp)
    # print(lateral_kp.multi_face_landmarks)

    front_lms = front_kp.multi_face_landmarks[0]
    lateral_lms = lateral_kp.multi_face_landmarks[0]

    # # draw landmarks on image for debug purpose
    # front_marked = np.copy(front)
    lateral_marked = np.copy(lateral)
    # mpDraw.draw_landmarks(front_marked, front_lms, {}, drawSpec, drawSpec)
    mpDraw.draw_landmarks(lateral_marked, lateral_lms, {}, drawSpec, drawSpec)

    # cv2.imshow('front_marked', front_marked)
    # cv2.waitKey(0)
    cv2.imshow('lateral_marked', lateral_marked)
    cv2.imwrite('landmarks_lateral.jpeg', lateral_marked)
    cv2.waitKey(0)

    # better structure for landmarks coords
    # TODO: high changes this can be useless and/or can be improved by a lot
    front_lms = np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in front_lms.landmark
        ])
    lateral_lms = np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in lateral_lms.landmark
        ])

    # calculate lateral face orientation
    # TODO: a better calculation would be based on the distance between landmark
    # 0 and 132 and 361
    nose_left_dist = np.abs(lateral_lms[132, 0] - lateral_lms[0, 0])
    nose_right_dist = np.abs(lateral_lms[361, 0] - lateral_lms[0, 0])
    orientation = 'left' if nose_left_dist < nose_right_dist else 'right'

    # python array to np array, maybe this can be improved
    c_src = []
    c_dst = []

    for i, _ in enumerate(lateral_lms):
        if i not in left_face: continue
        c_src.append([lateral_lms[i, 0], lateral_lms[i, 1]])
        c_dst.append([front_lms[i, 0], front_lms[i, 1]])

    c_src = np.array(c_src)
    c_dst = np.array(c_dst)

    cropped = crop_image(lateral, c_src)

    warped = trasformation_image(front, cropped, c_src, c_dst)
    return warped

if __name__ == "__main__":
    # Read front image and lateral image
    front = cv2.imread('test_images/1_front.jpg')
    lateral = cv2.imread('test_images/1_lat.jpg')

    warped = warp_face(front, lateral)

    cv2.imshow('warped', warped)
    cv2.waitKey(0)
