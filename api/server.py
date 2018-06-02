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
from manager import InferenceConfig, TFManager

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.numberplate import numberplate

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Path to pre-trained weights
NUMBERPLATE_WEIGHTS_PATH = "../weights/"

############################################################
#  Apply Mask
#  Load every image from destination folder and apply mask then save images
############################################################

manager = TFManager(address=('127.0.0.1', 50000), authkey='hello'.encode('UTF-8'))

def run_detect(filename):

    base_file_name = os.path.basename(filename)
    base_dir_name = os.path.dirname(filename)
    logging.info('Dir name: %s',base_dir_name)
    logging.info('File name: %s',base_file_name)
    split_file_name, split_file_ext = os.path.splitext(base_file_name)
    saved_dir_name = 'masked'
    saved_file_name = str.join('\\', (base_dir_name, saved_dir_name, base_file_name))
    logging.info('Saved as: %s',saved_file_name)

    # Convert png with alpha channel with shape[2] == 4 into shape[2] ==3 RGB images
    image = skimage.io.imread(filename)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Run detection
    results = model.detect([image], verbose=1)
    r= results[0]

    # Just apply mask then save images
    print_img = visualize.apply_mask_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    skimage.io.imsave(saved_file_name,print_img)

    return True

TFManager.register("run_detect", run_detect)

if __name__ == "__main__":

    config = InferenceConfig()
    config.display()

    # Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'numberplate']

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load pre-trained weights
    NUMBERPLATE_WEIGHTS_FILE = os.path.join(NUMBERPLATE_WEIGHTS_PATH, "mask_rcnn_numberplate.h5")
    model.load_weights(NUMBERPLATE_WEIGHTS_FILE, by_name=True)

    # VERY VERY IMPORTANT
    # https://github.com/matterport/Mask_RCNN/issues/600#issuecomment-393142704
    model.keras_model._make_predict_function()

    print('Weights file loaded.')

    server = manager.get_server()
    server.serve_forever()
