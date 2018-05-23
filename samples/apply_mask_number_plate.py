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


############################################################
#  Apply Mask
#  Load every image from destination folder and apply mask then save images
############################################################
def apply_mask_to_image(config_name,weights_name):

    # Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'numberplate']

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_name)

    # Load pre-trained weights
    NUMBERPLATE_WEIGHTS_FILE = os.path.join(NUMBERPLATE_WEIGHTS_PATH, weights_name)
    model.load_weights(NUMBERPLATE_WEIGHTS_FILE, by_name=True)
    logging.info('LOGGING:::::: %s weights file loaded.'%args.weights)

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
                logging.info('LOGGING:::::: Counting %d start' % file_count)

                # Date time log
                img_timestamp = time.localtime()
                date_string = str.join('/',(str(img_timestamp.tm_year), str(img_timestamp.tm_mon).zfill(2), str(img_timestamp.tm_mday).zfill(2)))
                time_string = str.join(':',(str(img_timestamp.tm_hour).zfill(2),str(img_timestamp.tm_min).zfill(2),str(img_timestamp.tm_sec).zfill(2)))
                logging.info('LOGGING:::::: Date Time: %s %s', date_string, time_string)

                # Filename and filepath log
                filename = os.path.join(IMAGE_DIR, file_names)
                logging.info('LOGGING:::::: Loading image: %s', file_names)

                base_file_name = os.path.basename(filename)
                base_dir_name = os.path.dirname(filename)
                logging.info('LOGGING:::::: Dir name: %s',base_dir_name)
                logging.info('LOGGING:::::: File name: %s',base_file_name)
                split_file_name, split_file_ext = os.path.splitext(base_file_name)
                saved_dir_name = 'masked'
                saved_file_name = str.join('\\', (base_dir_name, saved_dir_name, base_file_name))
                logging.info('LOGGING:::::: Saved as: %s',saved_file_name)

                # Convert png with alpha channel with shape[2] == 4 into shape[2] ==3 RGB images
                image = skimage.io.imread(filename)
                if len(image.shape) > 2 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                # Processing time log
                t0 = time.perf_counter()

                # Run detection
                results = model.detect([image], verbose=1)
                r = results[0]

                # Without displaying images for batch program
                ###logging.info_img = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                ###skimage.io.imsave(saved_file_name,logging.info_img)

                # Just apply mask then save images
                logging.info_img = visualize.apply_mask_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                skimage.io.imsave(saved_file_name,logging.info_img)

                # Processing time log
                t1 = time.perf_counter()
                logging.info('LOGGING:::::: Completed 1 image in %f sec',(t1-t0))
                logging.info('LOGGING:::::: End')
                # End

############################################################
#  __main__
#  Parse .h5 file as weight_name
#  python apply_mask_number_plate.py --weights=mask_rcnn_numberplate.h5
############################################################
if __name__ == '__main__':
    # Logging confg
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Apply masks to detected number plates.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    args = parser.parse_args()

    # Validate arguments
    if args.weights == None:
        logging.info('FATAL ERROR:::::: %s weights file not found.', args.weights)
    else:
        config = InferenceConfig()
        config.display()

        apply_mask_to_image(config, args.weights)
