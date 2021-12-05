# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:45:39 2021

@author: Divy
"""
import datetime
import logging
import time
from model import init_model
log_file = "Training_logs.txt"
if not log_file:
    f = open(log_file, 'x')
    f.close()
logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='a')
logger = logging.getLogger(__name__)


def job():
    try:
        logger.debug(f"{datetime.datetime.now()}: Training started")
        start = time.time()
        accuracy = init_model()
        if accuracy is None:
            logger.debug(f"{datetime.datetime.now()}: Not adequate data for training")
        logger.debug(f"{datetime.datetime.now()}: Training ended in {str(datetime.timedelta(seconds = (time.time() - start)))}.")
        logger.debug(f"{datetime.datetime.now()}: Train Accuracy: {accuracy['training']*100}%")
        logger.debug(f"{datetime.datetime.now()}: Validation Accuracy: {accuracy['validation']*100}%")
        logger.debug("")
    except Exception as e:
        logger.debug(f"{datetime.datetime.now()}: Exception occured in Training -> {e}")
    return


if __name__ == "__main__":
    m = 0
    s = 0
    ms = 0
    ttr = [0]
    while True:
        time_ = datetime.datetime.now()
        if (time_.hour in ttr) and time_.minute == 0 and time_.second == 0 and time_.microsecond == 0:
            job()
