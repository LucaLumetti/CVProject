#!/usr/bin/python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

fixed_landmarks = [234, 454, 447, 345, 346, 347, 348, 349, 277, 437, 399, 419,
        197, 196, 174, 47, 120, 119, 118, 117, 116, 34, 127]

chin_landmark = 152
leftcheek_landmark = 234
rigthcheek_landmark = 356

def get_image(pathname):
    img = cv2.imread(pathname)
    img_height, img_width, img_channels = img.shape
    img_area = img_height*img_width
    img = resize_with_ratio(img,height=720)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    return img

def resize_with_ratio(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))

    return cv2.resize(img, dim, interpolation=inter)

# keypoint detection
def find_facial_landmarks(img, landmarks=[], debug=False):

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

        xtl, ytl = int(faceLms.landmark[leftcheek_landmark].x*iw ), int(faceLms.landmark[leftcheek_landmark].y*ih)
        xbr, ybr = int(faceLms.landmark[rigthcheek_landmark].x*iw )+30, int(faceLms.landmark[chin_landmark].y*ih )+30

        coordinates = (xtl,ytl,xbr-xtl,ybr-ytl)

    if debug:
        cv2.imshow('landmarks', img_land)
        cv2.waitKey(0)

    return coordinates

# Color quantization on img with fixed number of bins
def color_quantization(img, bins=2, debug=False):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, bins, None,
            criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    count_color = [0 for _ in range(bins)]

    for elem in label:
        count_color[elem[0]] += 1
    index_list = np.argsort(count_color)
    reference = center[index_list[-1]]
    if (reference == [0, 0, 0]).all():
       reference = center[index_list[-2]]

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    if debug:
        cv2.imshow('kmeans', res2)
        cv2.waitKey(0)
    return res2, reference

'''
 Function that given an img return a binary mask (np.array) of the surgical
 mask detected in the image img. If facial landmarks are not detected, return
 an empty np.array.
'''
def find_mask(img, debug=False):

    # Keypoints detection
    coordinates = find_facial_landmarks(img, debug=debug)
    if not coordinates:
        return np.array([])
    # Creating mask to isolate surgical mask area
    mask = np.zeros(img.shape[:2], np.uint8)

    # Grabcut
    fgModel = np.zeros((1,65),np.float64)
    bgModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img,mask,coordinates,bgModel,fgModel,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
    output_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD),0,1).astype('uint8')
    output_mask *= 255
    img = img*output_mask[:,:,np.newaxis]

    # hsv
    out = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)

    # Apply kmeans to find dominant color
    res, reference = color_quantization(hsv, bins=3, debug=debug)

    # Thresholding
    res = res[:, :, 2]
    hist = cv2.calcHist([res], [0], output_mask, [256], [0, 256])
    max_value_index = np.argmax(hist)
    if max_value_index < 128:
        res[res == max_value_index] = 255
        max_value_index = 255
    offset = 5
    ret, thresh = cv2.threshold(res, max_value_index - offset,
            255, cv2.THRESH_BINARY)

    thresh //= 255

    # Closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Erosion
    eroded = cv2.erode(closing, kernel, iterations=5)

    # dilation
    dilated = cv2.dilate(eroded, kernel, iterations=10)

    if debug:
        cv2.imshow('polygon', mask)
        cv2.waitKey(0)
        cv2.imshow('mask', output_mask)
        cv2.waitKey(0)
        cv2.imshow('blur', out)
        cv2.waitKey(0)
        cv2.imshow('hsv to gray', res)
        cv2.waitKey(0)
        plt.subplot(221)
        plt.plot(hist)
        plt.show()
        cv2.imshow('threshold', thresh)
        cv2.waitKey(0)
        cv2.imshow('closing', closing)
        cv2.waitKey(0)
        cv2.imshow('eroded', eroded)
        cv2.waitKey(0)
        cv2.imshow('dilated', dilated)
        cv2.waitKey(0)

    return dilated


if __name__ == '__main__':

    # Read image
    img = get_image('./masked_people/2.jpeg')

    # Find mask
    mask = find_mask(img)
    if mask.size == 0:
        print('[ERROR] Unable to detect facial landmarks')
        exit(1)

    cv2.imshow('Mask', mask)
    cv2.waitKey(0)



