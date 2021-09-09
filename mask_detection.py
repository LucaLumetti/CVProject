#!/usr/bin/python3
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from itertools import chain
import mediapipe as mp

fixed_landmarks = [234, 454, 447, 345, 346, 347, 348, 349, 277, 437, 399, 419, 197, 196, 174, 47, 120, 119, 118, 117,
                   116, 34, 127]

def get_image(pathname):
    img = cv2.imread(pathname)
    # img = cv2.imread('../front_lateral_people/1_front2.jpg')
    img_height, img_width, img_channels = img.shape
    img_area = img_height*img_width
    img = ResizeWithRatio(img,width=720)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    return img

def ResizeWithRatio(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img
    if width is None:
        r = height/float(h)
        dim = (int(w*r),height)
    else:
        r = width /float(w)
        dim = (width,int(h*r))

    return cv2.resize(img,dim,interpolation=inter)

# keypoint detection
def find_facial_landmarks(img,landmarks=[]):

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
            mpDraw.draw_landmarks(img_land, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            #for id,lm in enumerate(faceLms.landmark):
            #    print(lm)
            #    ih, iw, ic = img.shape
            #    x,y = int(lm.x*iw), int(lm.y*ih)
            #    print(id, x,y)
        ih,iw,ic = img.shape
        #xc1,yc1 = int(faceLms.landmark[18].x*iw),int(faceLms.landmark[18].y*ih)
        #xc2,yc2 = int(faceLms.landmark[135].x*iw),int(faceLms.landmark[135].y*ih)
        #xc3, yc3 = int(faceLms.landmark[376].x * iw), int(faceLms.landmark[376].y * ih)
        #xtl,ytl,xbr,ybr = int(faceLms.landmark[127].x*iw),int(faceLms.landmark[127].y*ih),int(faceLms.landmark[356].x*iw),int(faceLms.landmark[152].y*ih)

        for landmark in landmarks_list:
            xc,yc =  int(faceLms.landmark[landmark].x*iw),int(faceLms.landmark[landmark].y*ih)
            if not landmark in fixed_landmarks:
                yc += 40
            else:
                yc -= 10
            #cv2.circle(img,(xc,yc),1,(0,255,0),3)
            keypoints.append((xc,yc))
    cv2.imshow('landmarks', img_land)
    cv2.waitKey(0)
    return keypoints


#cv2.drawContours(img_land,np.array([keypoints]),0,(0,255,0),2)

def color_quantification(hsv):
    Z = hsv.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    count_color = [0 for _ in range(K)]

    for elem in label:
        count_color[elem[0]] += 1
    index_list = np.argsort(count_color)
    reference = center[index_list[-1]]
    if (reference == [0,0,0]).all():
       reference = center[index_list[-2]]
    print('dominant color is: '+str(reference))
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imshow('kmeans',res2)
    cv2.waitKey(0)
    return res2, reference



if __name__ == '__main__':

    # Read image
    img = get_image('./masked_people/3.jpeg')

    # Keypoints detection
    keypoints = find_facial_landmarks(img)

    # Grayscaling
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)

    # Creating mask to isolate surgical mask area
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.imshow('polygon', cv2.fillPoly(mask, np.array([keypoints]), (255, 255, 255)))
    #mask[ytl:ybr+30,xtl:xbr+30] = img[ytl:ybr+30,xtl :xbr+30]
    out = cv2.bitwise_and(img, img, mask=mask)
    #not_black_pxl = np.any(out != [0, 0, 0], axis=-1)
    cv2.imshow('mask', out)
    cv2.waitKey(0)

    # hsv
    img = cv2.GaussianBlur(out, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('blur', img)
    cv2.waitKey(0)

    # Apply kmeans to find dominant color
    res,reference = color_quantification(hsv)

    # Thresholding
    #reference = np.round(np.mean(np.array([hsv[yc1,xc1],hsv[yc3,xc3],hsv[yc3,xc3]]),axis=0))
    #reference = hsv[yc1,xc1]
    #print(reference, reference - [20, 30, 30], reference + [20, 30, 30])
    thresh = cv2.inRange(res, reference - [20, 30, 30], reference + [20, 30, 30])
    #ret,thresh = cv2.threshold(cv2.cvtColor(out,cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('threshold', thresh)
    cv2.waitKey(0)

    kernel = np.ones((2, 2), np.uint8)
    # Erosion
    #eroded = cv2.erode(thresh, kernel, iterations=5)
    #cv2.imshow('eroded', eroded)
    #cv2.waitKey(0)

    # medblur
    medblur = cv2.medianBlur(thresh, 5)
    cv2.imshow('medblur', medblur)
    cv2.waitKey(0)

    # dilation
    dilated = cv2.dilate(medblur, kernel, iterations=15)
    cv2.imshow('dilated', dilated)
    cv2.waitKey(0)