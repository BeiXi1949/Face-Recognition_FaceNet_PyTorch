#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 6/29/18
# Author : ZOU Zijie
# Email  : zouzijie1994@gmail.com
# Plateform : pycharm

"""
This function is for reading traning set and output KNN result
-------------ZOU Zijie :)

"""


import os
import csv
import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier,DistanceMetric
from sklearn.externals import joblib
import sys
sys.path.append(os.path.realpath('.'))

#/------Initial Logger------/
logi_time = time.strftime("_%d-%b-%Y_%H:%M:%S", time.localtime())



def falttern(path):
    vector=np.load(path)
    result = vector.flatten()

    return result



def generate_dataset(csv_path):
    x_train=[]
    y_train=[]

    for root, dirs, files in os.walk(csv_path):
        pass
    # print(root)
    # print(files)

    for j in files:
        csv_temp = root+"/"+j
        name = j[:-4]

        rf = open(csv_temp, 'r')
        reader = list(csv.reader(rf))
        counter =0
        for k in reader:
            if counter>0:
                read_path = k[3]
                data = falttern(read_path)
                x_train.append(data)
                y_train.append(name)
            else:
                pass
            counter+=1
    return x_train,y_train


def knn_classifier(input,knn_path):
    knn = joblib.load(knn_path)

    prob = knn.predict_proba(input)
    pred = knn.predict(input)
    # print(max(distance[0][0]),pred)
    return pred,prob

def training_KNN(csv_path):
    X_train,Y_train=generate_dataset(csv_path)
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    knn.fit(X_train, Y_train)
    joblib.dump(knn, './models/knn.model')
    # return print("---KNN model saved---")

