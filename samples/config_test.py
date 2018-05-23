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

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.calpis import calpis

# matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
CALPIS_WEIGHTS_PATH = "../mask_rcnn_calpis water.h5"  # TODO: update this path

class InferenceConfig(calpis.CalpisConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(CALPIS_WEIGHTS_PATH, by_name=True)

# Calpis Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'calpis']



# Load every image from the images folder

# image_type_ok_list = ['jpeg','png','gif','bmp']
image_type_ok_list = ['jpeg']

for file_names in os.scandir(IMAGE_DIR):
    if file_names.is_file():
        image_type = imghdr.what(file_names)
        if image_type in image_type_ok_list:
            print('LOGGING:::::: Start')
            filename = os.path.join(IMAGE_DIR, file_names)
            print('LOGGING:::::: Loading image:',file_names)

            image = skimage.io.imread(filename)
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            base_file_name = os.path.basename(filename)
            base_dir_name = os.path.dirname(filename)
            print('LOGGING:::::: Dir name:',base_dir_name)
            print('LOGGING:::::: File name:',base_file_name)
            split_file_name, split_file_ext = os.path.splitext(base_file_name)

            img_timestamp = time.localtime()
            time_string = str.join('/',(str(img_timestamp.tm_year), str(img_timestamp.tm_mon).zfill(2), str(img_timestamp.tm_mday).zfill(2)))
            print('LOGGING:::::: Date:',time_string)

            saved_dir_name = 'masked'
            saved_file_name = str.join('\\', (base_dir_name, saved_dir_name, base_file_name))
            print('LOGGING:::::: Saved as:',saved_file_name)

            t0 = time.perf_counter()
            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
            #                            class_names, r['scores'])
            #print_img = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            #skimage.io.imsave(saved_file_name,print_img)
            print_img = visualize.apply_mask_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            skimage.io.imsave(saved_file_name,print_img)
            t1 = time.perf_counter()
            print('LOGGING:::::: Completed in %f sec'%(t1-t0))
            print('LOGGING:::::: End\n\n\n')
