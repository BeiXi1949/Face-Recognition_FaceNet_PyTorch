#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 6/29/18
# Author : ZOU Zijie
# Email  : zouzijie1994@gmail.com
# Plateform : pycharm

import time
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
from imutils import face_utils
import dlib
try:
    from .utils.openface_pytorch import netOpenFace
except:
    from utils.openface_pytorch import netOpenFace

from KNN_builder import  knn_classifier

try:
    from .utils.operations import prewhiten
except:
    from utils.operations import prewhiten

#/------Initial Logger------/
logi_time = time.strftime("_%d-%b-%Y_%H:%M:%S", time.localtime())


#/---Dlib Initial---/
model = './models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=250)

def round_up(value):

    return round(value * 100) / 100.0

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def prepare_openface(useCuda=False, gpuDevice=0, useMultiGPU=False):
    model = netOpenFace(useCuda, gpuDevice)
    # model.load_state_dict(torch.load(./models/openface_nn4_small2_v1.pth'))
    model.load_state_dict(torch.load('./models/openface_20180119.pth',map_location=lambda storage, loc: storage))

    if useMultiGPU:
        model = nn.DataParallel(model)

    return model

if __name__ == '__main__':
    knn_path = './models/knn.model'
    video_capture = cv2.VideoCapture(0)
    frame_index = 0

    # /---Loading Pytotch Model---/
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    facenet = prepare_openface()

    if torch.cuda.is_available():
        facenet.cuda()

    # /---Main---/
    while (video_capture.isOpened()):
        total_s = time.time()

        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b_box = []

        try:
            detect_s = time.time()

            rects = detector(gray, 1)

            detect_end = time.time()
            mainround_s = time.time()
            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                align_s = time.time()

                faceAligned = fa.align(frame, gray, rect)
                image = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
                # print(image)
                # image = prewhiten(image)
                pil_im = Image.fromarray(image)
                align_e = time.time()

                # /---compute vectors and do KNN---/
                pyt_s = time.time()
                img_tensor = transform(pil_im)
                img_tensor = to_var(img_tensor)
                outputs_128, outputs_726 = facenet(img_tensor.unsqueeze(0))
                outputs = to_np(outputs_128)

                outputs = outputs.flatten().reshape(1, -1)

                pyt_e = time.time()
                # /---Multiprocessing for KNN---/
                knn_s = time.time()
                pred, prob = knn_classifier(outputs, knn_path)

                name = pred[0]
                knn_e = time.time()
                b_box.append(((x, y), (x + w, y + h), name))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_TRIPLEX
                cv2.putText(frame, name, (x, y), font, 1, (255, 255, 0), 1, True)

            mainround_e = time.time()

            # /---print time---/
            # print("mainround_time=", mainround_e - mainround_s)
            # print("detect_time=", detect_end - detect_s)
            # print("align_time=", align_e - align_s)
            # print("facenet=", pyt_e - pyt_s)
            # print("knn=", knn_e - knn_s)

            print("Frame=" + str(frame_index) + "---" + name)
            Intro = "Press 'ESC' to quit"
            cv2.putText(frame, Intro, (40, 40), font, 0.5, (0, 0, 255), 0, True)
            cv2.imshow("Rec Recoginition", frame)
            total_e = time.time()
            print("total=", total_e - total_s)

            frame_index += 1

            if cv2.waitKey(40) & 0xFF == 27:
                break

        except:
            font = cv2.FONT_HERSHEY_TRIPLEX
            Intro = "Press 'ESC' to quit"
            cv2.putText(frame, Intro, (40, 40), font, 0.5, (0, 0, 255), 0, True)
            Warnning = "No registration!!!"
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame, Warnning, (40, 80), font, 1, (0, 0, 255), 1, True)
            cv2.imshow("Rec Recoginition", frame)

            if cv2.waitKey(40) & 0xFF == 27:
                break

    cv2.destroyAllWindows()