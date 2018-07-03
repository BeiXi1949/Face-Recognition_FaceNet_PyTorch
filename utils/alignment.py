#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 6/29/18
# Author : ZOU Zijie
# Email  : zouzijie1994@gmail.com
# Plateform : pycharm

import sys
import os
from PIL import Image
import numpy as np
sys.path.append(os.path.realpath('.'))
from src import detect_faces, show_bboxes
import cv2
from imutils import face_utils
import dlib

'''
Global variables
'''

model = './models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

def generate_landmarks(base_path,new_img):
    base_img = Image.open(base_path)
    new_img =  cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(new_img)

    b_point=[]
    n_point=[]

    bounding_b_base,landmarks_base = detect_faces(base_img)
    bounding_b_new, landmarks_new = detect_faces(pil_im)
    for i in range (2):
        b_point.append((landmarks_base[0][i],landmarks_base[0][i+5]))
        n_point.append((landmarks_new[0][i], landmarks_new[0][i + 5]))

    b_point.append((landmarks_base[0][3], landmarks_base[0][3 + 5]))
    n_point.append((landmarks_new[0][3], landmarks_new[0][3 + 5]))

    return np.array(b_point),np.array(n_point)

def alignment_trans(b_point,n_point,img):
    rows,cols,ch=img.shape
    pts1 = b_point
    pts2 = n_point

    M = cv2.getAffineTransform(pts2,pts1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst

def align(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):

        faceAligned = fa.align(img, gray, rect)
        faceAligned = cv2.resize(faceAligned, (250, 250))

    return faceAligned

