#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 6/29/18
# Author : ZOU Zijie
# Email  : zouzijie1994@gmail.com
# Plateform : pycharm


import sys
# sys.path.append('/home/zouzijie/Desktop/Particial_Facenet/FaceRec_Pytorch_V1.3')
import time
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from multiprocessing import Pool
import csv
import multiprocessing as mp
from multiprocessing import Process
from imutils import face_utils
import dlib


try:
    from .utils.openface_pytorch import netOpenFace
except:
    from utils.openface_pytorch import netOpenFace
try:
    from .utils.generate_csv import write_Users_csv,write_Name_csv,read_csv,write_vector_csv
except:
    from utils.generate_csv import write_Users_csv,write_Name_csv,read_csv,write_vector_csv
try:
    from .KNN_builder import training_KNN,generate_dataset
except:
    from KNN_builder import training_KNN,generate_dataset

try:
    from .utils.operations import prewhiten
except:
    from utils.operations import prewhiten


sys.path.append(os.path.realpath('.'))


# print(sys.path)
"""

This function is only  for this Face Recoginition System.
if you want to import to another functions please modify

---------Created by ZOU Zijie :)

"""

#/------Initial Logger------/
log_timee = time.strftime("_%d-%b-%Y_%H:%M:%S", time.localtime())


#/---Dlib initial---/
model = './models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=250)

def log_time():
    #/--1 for info; 2 for warning; 3 for error--/
    log_time = time.strftime("%d-%b-%Y_%H:%M:%S", time.localtime())

    return log_time


def align_unknown():
    user = "Unknown"
    path = './Users/people_ori/Unkonwn'
    un_vec = './Users/people_vectors/Unkonwn'

    count = 0

    folder = "./Users/people_ali/" + user
    if not os.path.exists(folder):
        os.mkdir(folder)

        for file in os.listdir(path):
            img_path = path + '/' + file
            ali_path = './Users/people_ali/Unknown' + '/' + file
            frame = cv2.imread(img_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):

                faceAligned = fa.align(frame, gray, rect)

            cv2.imwrite(ali_path, faceAligned)

            write_Name_csv(count,user,Ori_path=img_path,Ali_path=ali_path)
            count+=1

    print("Unknown aligned and csv is prepared")

    print("checking unknown vectors")
    if os.path.exists(un_vec) is False:
        os.mkdir(un_vec)
        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        facenet = prepare_openface()

        if torch.cuda.is_available():
            facenet.cuda()

        result, length, dir = write_vector_csv(user)
        counter1 = 0

        for i in result:
            if counter1==0:
                pass
            else:
                temp_path = i[2]
                bValid = True
                try:
                    img_pil = Image.open(temp_path)
                    img_tensor = transform(img_pil)
                    img_tensor = to_var(img_tensor)
                    outputs = facenet(img_tensor.unsqueeze(0))
                    if counter1 < 10:
                        vector_temp = un_vec + '/' + user + '_000' + str(counter1) + '.npy'
                        np.save(vector_temp, to_np(outputs[0]))
                        i[3] = vector_temp
                        wf1 = open(dir, 'w')
                        writer1 = csv.writer(wf1)
                        writer1.writerows(result)
                    elif 10 <= counter1 < 100:
                        vector_temp = un_vec + '/' + user + '_00' + str(counter1) + '.npy'
                        np.save(vector_temp, to_np(outputs[0]))
                        i[3] = vector_temp
                        wf1 = open(dir, 'w')
                        writer1 = csv.writer(wf1)
                        writer1.writerows(result)
                    elif 100 <= counter1 < 1000:
                        vector_temp = un_vec + '/' + user + '_0' + str(counter1) + '.npy'
                        np.save(vector_temp, to_np(outputs[0]))
                        i[3] = vector_temp
                        wf1 = open(dir, 'w')
                        writer1 = csv.writer(wf1)
                        writer1.writerows(result)

                    # counter1 += 1

                    wf1.close()

                except:
                    bValid = False
                    result.pop[counter1]
                    counter1 -= 1
                    print("Find error img! Delete")
                    print("Error path drop")

            counter1 += 1

        print("/----Finished----/")
    else:
        print("/----Finished----/")


def write_user(v_s_path,counter,origin,folder,person):
    capture = cv2.VideoCapture(v_s_path)


    while counter < 150:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            faceAligned = fa.align(frame, gray, rect)
            if counter < 10:
                ori_path = origin + '/' + person + '_000' + str(counter) + '.jpg'
                ali_path = folder + '/' + person + '_000' + str(counter) + '.jpg'
                # cv2.imwrite(ori_path, frame)
                cv2.imwrite(ali_path, faceAligned)
                write_Name_csv(counter, person, ori_path, ali_path)
            elif 10 <= counter < 100:
                ori_path1 = origin + '/' + person + '_00' + str(counter) + '.jpg'
                ali_path1 = folder + '/' + person + '_00' + str(counter) + '.jpg'
                # cv2.imwrite(ori_path1, frame)
                cv2.imwrite(ali_path1, faceAligned)
                write_Name_csv(counter, person, ori_path1, ali_path1)
            elif 100 <= counter < 1000:
                ori_path2 = origin + '/' + person + '_0' + str(counter) + '.jpg'
                ali_path2 = folder + '/' + person + '_0' + str(counter) + '.jpg'
                # cv2.imwrite(ori_path2, frame)
                cv2.imwrite(ali_path2, faceAligned)
                write_Name_csv(counter, person, ori_path2, ali_path2)

            counter += 1
        print("Captured image : " + person + str(counter))





def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def prepare_openface(useCuda=False, gpuDevice=0, useMultiGPU=False):
    model = netOpenFace(useCuda, gpuDevice)
    # model.load_state_dict(torch.load('/home/zouzijie/Desktop/Particial_Facenet/FaceRec_PyTorch_V1.3/models/openface_nn4_small2_v1.pth'))
    model.load_state_dict(torch.load('./models/openface_20180119.pth',map_location=lambda storage, loc: storage))

    if useMultiGPU:
        model = nn.DataParallel(model)

    return model

def train_knn(path):
    l.acquire()
    # print("---Start training KNN---")
    training_KNN(path)
    # print("---saved KNN model---")
    l.release()

if __name__ == '__main__':
    p = Pool(processes=mp.cpu_count())
    print(mp.cpu_count())

    video_capture = cv2.VideoCapture(0)
    person = input('Person: ')
    origin = "./Users/people_ori/" + person
    folder = "./Users/people_ali/" + person
    vector_path = "./Users/people_vectors/"+person
    csv_path='./Users/csv/indiv'
    video_path='./Users/Videos/'

    write_Users_csv(person)

    try:
        if not os.path.exists(video_path):
            os.mkdir(video_path)

        if not os.path.exists(origin):
            os.mkdir(origin)
            os.mkdir(folder)
            os.mkdir(vector_path)

            record = 0
            counter = 0
            timer = 0

            r_s = time.time()
            ret, frame_t = video_capture.read()
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            FrameSize = (frame_t.shape[1], frame_t.shape[0])
            v_s_path = video_path + '/' + person + '.avi'
            outfile = cv2.VideoWriter(v_s_path, fourcc, 30., FrameSize)

            while (video_capture.isOpened()):
                ret, frame = video_capture.read()
                font = cv2.FONT_HERSHEY_TRIPLEX
                Intro = "adjust, 'a' to start,'ESC' to quit"
                cv2.putText(frame, Intro, (40, 40), font, 0.5, (255, 0, 0), 0, True)
                cv2.imshow('Capture', frame)

                if cv2.waitKey(50) & 0xFF == ord('a'):
                    while True:
                        record +=1
                        ret, frame = video_capture.read()
                        outfile.write(frame)
                        Intro1 = "shake your head, wait 10 seconds to quit"
                        cv2.putText(frame, Intro1, (40,40), font, 0.5, (255, 0, 0), 0, True)
                        cv2.imshow('Capture', frame)
                        cv2.waitKey(1)
                        if record == 180:
                            break

                if cv2.waitKey(40) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    break

            video_capture.release()
            outfile.release()
            cv2.destroyAllWindows()
            print("/------Finished Recording------/")

            p.apply_async(write_user, args=(v_s_path, counter, origin, folder, person,))
            p.close()
            p.join()
            print("/------Finished Reigister------/")
            os.remove(v_s_path)

        else:
            print("This name already exists.")
            sys.exit("sorry, goodbye!")
    except os.error:
        print("Making dir error")


    print('\n'+'/------Start Processing Vectors------/')
    time.sleep(1)
    cv2.destroyAllWindows()

    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    facenet = prepare_openface()

    if torch.cuda.is_available():
        facenet.cuda()

    result,length,dir = write_vector_csv(person)
    counter1 =0

    for i in result:
        if counter1==0:
            pass
        else:
            temp_path=i[2]
            bValid = True
            try:
                img_pil = Image.open(temp_path)
                img_tensor = transform(img_pil)
                img_tensor = to_var(img_tensor)
                outputs = facenet(img_tensor.unsqueeze(0))
                if counter1 < 10:
                    vector_temp = vector_path + '/' + person + '_000' + str(counter1) + '.npy'
                    np.save(vector_temp, to_np(outputs[0]))
                    i[3]=vector_temp
                    wf1 = open(dir, 'w')
                    writer1 = csv.writer(wf1)
                    writer1.writerows(result)
                elif 10 <= counter1 < 100:
                    vector_temp = vector_path + '/' + person + '_00' + str(counter1) + '.npy'
                    np.save(vector_temp, to_np(outputs[0]))
                    i[3] = vector_temp
                    wf1 = open(dir, 'w')
                    writer1 = csv.writer(wf1)
                    writer1.writerows(result)
                elif 100 <= counter1 < 1000:
                    vector_temp = vector_path + '/' + person + '_0' + str(counter1) + '.npy'
                    np.save(vector_temp, to_np(outputs[0]))
                    i[3] = vector_temp
                    wf1 = open(dir, 'w')
                    writer1 = csv.writer(wf1)
                    writer1.writerows(result)

                # counter1 += 1

                wf1.close()

            except:
                bValid = False
                result.pop[counter1]
                counter1 -=1
                print("Find error img! Delete")
                print("Error path drop")

        counter1 += 1

    print("/----Finished----/")

    time.sleep(1)
    print("check loading unknown data")
    align_unknown()
    l = mp.Lock()
    p= Process(target=train_knn, args=(csv_path,))
    p.start()
    p.join()



