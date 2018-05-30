# Import libraries
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imghdr
import cv2
import skimage.io
import logging
import argparse

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
from manager import InferenceConfig,TFManager

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")




############################################################
#  Apply Mask
#  Load every image from destination folder and apply mask then save images
############################################################
if __name__ == "__main__":
    manager = TFManager(address=('127.0.0.1', 50000), authkey='hello'.encode('UTF-8'))
    manager.connect()

    TFManager.register("run_detect")

    # Validation images extension
    image_type_ok_list = ['jpeg','png','gif','bmp']

    # Loop every file in the folder
    file_count = 0
    for file_names in os.scandir(IMAGE_DIR):
        if file_names.is_file():
            image_type = imghdr.what(file_names)
            if image_type in image_type_ok_list:
                # Start
                file_count +=1
                print('Counting %d start' % file_count)

                # Filename and filepath log
                filename = os.path.join(IMAGE_DIR, file_names)

                r = manager.run_detect(filename)
                print(r)

                # Without displaying images for batch program
                ###logging.info_img = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                ###skimage.io.imsave(saved_file_name,logging.info_img)
                                
                print('End')
                # End
