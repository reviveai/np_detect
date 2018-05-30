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

# Import flask libraries
import hashlib
import json
from time import time
from urllib.parse import urlparse
from uuid import uuid4

import requests
from flask import Flask, jsonify, request

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
#  Configurations
#  inherits from config.py
############################################################

class InferenceConfig(numberplate.NumberPlateConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

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


def run_detect(filename):

    base_file_name = os.path.basename(filename)
    base_dir_name = os.path.dirname(filename)
    split_file_name, split_file_ext = os.path.splitext(base_file_name)
    saved_dir_name = 'masked'
    saved_file_name = str.join('\\', (base_dir_name, saved_dir_name, base_file_name))

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


# Instantiate the Node
app = Flask(__name__)


@app.route('/mine', methods=['GET'])
def mine():
    filename = os.path.join(IMAGE_DIR, 'np_train (46).jpg')

    r = run_detect(filename)

    response = {
        'message': "done",
    }
    return jsonify(response), 200


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='0.0.0.0', port=port)
