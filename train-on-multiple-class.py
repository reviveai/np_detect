
"""
train on multiple classes sample
check origin file
https://github.com/matterport/Mask_RCNN/blob/v2.1/samples/balloon/balloon.py

origin issues
https://github.com/matterport/Mask_RCNN/issues/372#issuecomment-388732088
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class NumberPlateConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "numberplate"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + numberplate

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################
class NumberPlateDataset(utils.Dataset):
    def load_multi_number(self, dataset_dir, subset):
    """Load a subset of the number dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    """
    # Add classes
    self.add_class("object", 1, "A")
    self.add_class("object", 2, "B")
    self.add_class("object", 3, "C")
    self.add_class("object", 4, "D")
    self.add_class("object", 5, "E")
    self.add_class("object", 6, "F")
    self.add_class("object", 7, "G")
    self.add_class("object", 8, "H")
    self.add_class("object", 9, "I")
    self.add_class("object", 10, "J")
    self.add_class("object", 11, "K")
    self.add_class("object", 12, "browl")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, ".../train/via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # for b in a['regions'].values():
            #    polygons = [{**b['shape_attributes'], **b['region_attributes']}]
            # print("string=", polygons)
            # for r in a['regions'].values():
            #    polygons = [r['shape_attributes']]
            #    # print("polygons=", polygons)
            #    multi_numbers = [r['region_attributes']]
                # print("multi_numbers=", multi_numbers)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            objects = [s['region_attributes'] for s in a['regions'].values()]   # 读取objects 从json
            # print("multi_numbers=", multi_numbers)
            # num_ids = [n for n in multi_numbers['number'].values()]
            # for n in multi_numbers:
            num_ids = [int(n['object']) for n in objects]  # 读取objects的class id (num_ids)
            # print("num_ids=", num_ids)
            # print("num_ids_new=", num_ids_new)
            # categories = [s['region_attributes'] for s in a['regions'].values()]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)  # 添加一个参数 把num ids 也加到图片dataset里


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a number dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # print("info['num_ids']=", info['num_ids'])
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)  # balloon模型里是默认1个class 默认是1 这里返回多个class id的num_ids
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
