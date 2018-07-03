#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 6/29/18
# Author : ZOU Zijie
# Email  : zouzijie1994@gmail.com
# Plateform : pycharm
import logging
import time
import sys
import os

sys.path.append(os.path.realpath('..'))

"""

This function is only suit for this Face Recoginition System for logging
if you want to import to another functions please modify

---------Created by ZOU Zijie :)

"""

class logger:
    def __init__(self):
        super(logger, self).__init__()
        log_time = time.strftime("_%d-%b-%Y_%H:%M:%S", time.localtime())
        self.file_name='./logs/register/register' + log_time + '.log'
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='/--%a, %d-%b-%Y, %H:%M:%S--/',
                            filename=self.file_name,
                            filemode='w')
        logging.debug('Start debuging')


    def info_logger(self,type):
        logger = logging.getLogger(self.file_name)
        if type=='INFO':
            logger.info('aa')
        elif type=='WARN':
            logger.warning('warn')
        elif type == 'ERROR':
            logger.error('error')

        return "Log Recoed"

