#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 6/29/18
# Author : ZOU Zijie
# Email  : zouzijie1994@gmail.com
# Plateform : pycharm

import os
import time

import os

# /---Return file names: root for present path; dirs for sub-dir; files for all doc but not dir---/
def file_name(file_dir):
    count=0
    for root, dirs, files in os.walk(file_dir):
        # print(count)
        # print("Root",root)
        # print("Dirs",dirs)
        # print("Files",files)
        # print('\n')
        if count ==0:
            with open('/home/zouzijie/Desktop/Particial_Facenet/FaceRec_Pytorch_V1.0/Users/class.txt', 'w') as w:
                for i in dirs:
                    w.write(i+'\t'+str(dirs.index(i))+'\n')
        elif count==1:
            total = int(len(files))
            print(total)
            with open('/home/zouzijie/Desktop/Particial_Facenet/FaceRec_Pytorch_V1.0/Users/dataset.txt', 'w') as e:
                for j in range(0,total-1):
                    for jj in range(j+1,total):
                        e.write(str(count-1)+'\t'+root+'/'+files[j]+'\t'+'\t'+root+'/'+files[jj]+'\n')
        else:
            total1 = int(len(files))
            with open('/home/zouzijie/Desktop/Particial_Facenet/FaceRec_Pytorch_V1.0/Users/dataset.txt', 'a') as e:
                for k in range(0, total1-1):
                    for kk in range(k+1, total1):
                        e.write(str(count-1)+'\t'+root +'/'+ files[k] + '\t'+'\t' + root+'/' + files[kk] + '\n')

        count+=1






# file_name('/home/zouzijie/Desktop/Particial_Facenet/FaceRec_Pytorch_V1.0/Users/people_ali')