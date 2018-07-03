#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 6/29/18
# Author : ZOU Zijie
# Email  : zouzijie1994@gmail.com
# Plateform : pycharm

import csv
import time
import os
import sys

"""

This function is only suit for this Face Recoginition System with logging
if you want to import to another functions please modify

---------Created by ZOU Zijie :)

"""
# sys.path.append(os.path.realpath('..'))
# print(sys.path)

#/------Initial Logger------/
logi_time = time.strftime("_%d-%b-%Y_%H:%M:%S", time.localtime())


def log_time():
    #/--1 for info; 2 for warning; 3 for error--/
    log_time = time.strftime("%d-%b-%Y_%H:%M:%S", time.localtime())

    return log_time

#/------Global Main Path------/
csv_dir = './Users/csv'
csv_dir_indiv='./Users/csv/indiv'
csv_dir_bl='./Users/csv/baseline'
def read_csv(ty,mode,name):

    if ty == 1:
        dir = csv_dir + '/Users.csv'
    elif ty == 2:
        dir = csv_dir_indiv + '/'+str(name)+'.csv'

    rf = open(dir, 'r')
    reader = list(csv.reader(rf))

    if mode == 'read':
        num = len(reader)
        return reader,num
    elif mode == 'check':
        num1 = len(reader)
        for i in reader:
            if name == i[1]:
                print("Warning: User is exsist!")
                result = False
                break
            else:
                result = True

        return result,num1

def write_Users_csv(User):
    try:
        result,num = read_csv(1,mode='check',name=User)
        if result == False:
            pass
        elif result == True:
            wf = open('./Users/csv/Users.csv', 'a')
            writer = csv.writer(wf)
            content = [num, User]
            writer.writerow(content)
            print("Write person success!" + str(User)+" at"+ str(log_time()))
    except:

        wf = open('./Users/csv/Users.csv', 'w')
        writer = csv.writer(wf)
        header = ["id","Name"]
        writer.writerow(header)
        unknown = ["1", "Unknown"]
        writer.writerow(unknown)
        content = ["2", User]
        writer.writerow(content)
        print("/===No csv...Creating...\n" + "/===Ceated!")

        # print("Write person error!"+str(User),"+ at"+str(log_time()))
        # logging.error("Write person error!"+str(User)+"+ at"+str(log_time()))


def write_Name_csv(num, User, Ori_path, Ali_path):
    dir = csv_dir_indiv + '/' + str(User) + '.csv'
    try:
        read_csv(2, mode='read', name=User)
        wf1 = open(dir, 'a')
        writer1 = csv.writer(wf1)
        header1 = [num,Ori_path, Ali_path, 'Null']
        writer1.writerow(header1)

        # print("Write img_path success!" + str(User) + " at" + str(log_time()))
    except:
        wf1 = open(dir, 'w')
        writer1 = csv.writer(wf1)
        header = ["No","Ori_img", "Ali_img", "Vectors"]
        writer1.writerow(header)
        content1 =  [num,Ori_path, Ali_path, 'Null']
        writer1.writerow(content1)

        print("/===No csv...Creating...\n" + "/===Ceated!")

def write_Unknown_csv(User):
    dir = csv_dir_indiv + '/' + str(User) + '.csv'
    try:
        path = './Users/people_ori/Unkonwn'
        alipath= './Users/people_ali/Unkonwn'
        num = 0
        for file in os.listdir(path):
            Ori_path = path+'/'+file
            Ali_path = alipath+'/'+file
            read_csv(2, mode='read', name=User)
            wf1 = open(dir, 'a')
            writer1 = csv.writer(wf1)
            header1 = [num,Ori_path, Ali_path, 'Null']
            writer1.writerow(header1)
            wf1.close()
            num+=1

        print("Write img_path success!" + str(User) + " at" + str(log_time()))
    except:
        path = './Users/people_ori/Unkonwn'
        alipath = './Users/people_ali/Unkonwn'
        num = 0
        for file in os.listdir(path):
            Ori_path = path + '/' + file
            Ali_path = alipath + '/' + file

            wf1 = open(dir, 'w')
            writer1 = csv.writer(wf1)
            header = ["No", "Ori_img", "Ali_img", "Vectors"]
            writer1.writerow(header)
            content1 = [num, Ori_path, Ali_path, 'Null']
            writer1.writerow(content1)
            wf1.close()
            num += 1

        print("/===No csv...Creating...\n" + "/===Ceated!")


def write_vector_csv(User):
    try:
        dir=csv_dir_indiv + '/' + str(User) + '.csv'
        result,length=read_csv(2, mode='read', name=User)
        print(("Write Vector success!" + str(User) + "at" + str(log_time())))
        return result, length, dir
    except:
        print(("Write Vector error!" + str(User) + "at" + str(log_time())))

def write_bl_csv(User):
    try:
        dir=csv_dir_bl + '/' + str(User) + '.csv'
        result,length=read_csv(2, mode='read', name=User)
        print(("Write Vector success!" + str(User) + "at" + str(log_time())))
        return result, length, dir
    except:
        print(("Write Vector success!" + str(User) + "at" + str(log_time())))
    pass

