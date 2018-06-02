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

# Directories
MODEL_DIR = os.path.join(ROOT_DIR, "logs")      #training logs(weights)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")    #images root folder
MASKED_DIR = os.path.join(IMAGE_DIR, "masked")  #masked images folder
UPLOAD_DIR = os.path.join(IMAGE_DIR, "upload")  #uploaed images folder

# Path to pre-trained weights
NUMBERPLATE_WEIGHTS_PATH = "../weights/"

# Logging confg
logging.basicConfig(level=logging.DEBUG, filename="log", filemode="a+",
                format="%(asctime)-15s %(levelname)-8s %(message)s")

############################################################
#  Configurations
#  Inherits from config.py
############################################################

class InferenceConfig(numberplate.NumberPlateConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 576

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

# VERY VERY IMPORTANT to pre load inference
# https://github.com/matterport/Mask_RCNN/issues/600#issuecomment-393142704
model.keras_model._make_predict_function()

logging.info('Model and weight have been loaded.')


def run_detect(filename):
    base_file_name = os.path.basename(filename)
    saved_file_name = os.path.join(MASKED_DIR, base_file_name)
    logging.info('Loading image: %s', base_file_name)

    # Convert png with alpha channel with shape[2] == 4 into shape[2] ==3 RGB images
    image = skimage.io.imread(filename)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    logging.info('Image shape info: %s', image.shape)

    # Run detection
    results = model.detect([image], verbose=1)
    r= results[0]
    logging.info('Runing model.detect([image], verbose=1).')

    # Just apply mask then save images
    print_img = visualize.apply_mask_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],None,None,None,None,None,[(1.0,1.0,1.0)])
    skimage.io.imsave(saved_file_name,print_img)
    logging.info('Finished apply_mask_instances.')

    return True


# Instantiate the Node
app = Flask(__name__)

@app.route('/applyMask', methods=['GET','POST'])
def apply_mask_get():

    # Validation images extension
    image_type_ok_list = ['jpeg','png','gif','bmp']

    if request.method == 'GET':

        image_name = request.args.get('image', default = '*', type = str)
        full_filename = os.path.join(IMAGE_DIR, image_name)
        image_type = imghdr.what(full_filename)

        if image_type in image_type_ok_list:
            r = run_detect(full_filename)
            response_msg = "Done. Applied mask into 1 image '%s'" % full_filename
            response = {
                'message': response_msg
            }
            return jsonify(response), 200
        else:
            response = {
                'message': "Invalid image type.(Allowed image type: JPEG, PNG, GIF, BMP)"
            }
            return jsonify(response), 400

    if request.method == 'POST':

        if 'file' not in request.files:
            response = {
                'message': "No file uploaded within the POST body."
            }
            return jsonify(response), 400

        uploaded_file = request.files['file']
        full_filename = os.path.join(UPLOAD_DIR, uploaded_file.filename)
        uploaded_file.save(full_filename)

        result = run_detect(full_filename)

        response_msg = "Done. Applied mask into 1 image '%s'" % full_filename
        response = {
            'message': response_msg
        }
        return jsonify(response), 200


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='0.0.0.0', port=port)
