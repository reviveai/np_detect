https://stackoverflow.com/questions/49684468/mask-r-cnn-for-object-detection-and-segmentation-train-for-a-custom-dataset

Following are the steps

You need to have all your annotations.
All of those need to be converted to VGG Polygon schema (yes i mean polygons). I have added a sample VGG Polygon format in the end of this answer.
You need to divide your custom dataset into train, test and val
The annotation by default are looked with a filename via_region_data.json inside the individual dataset folder. For eg for training images it would look at train\via_region_data.json. You can also change it if you want.
Inside Samples folder you can find folders like Balloon, Nucleus, Shapes etc. Copy one of the folders. Preferably balloon.
Inside the copied folder, you will have a .py file (for balloon it will be balloon.py), change the following variables
ROOT_DIR : the absolute path where you have cloned the project
DEFAULT_LOGS_DIR : This folder will get bigger in size so change this path accordingly (if you are running your code in a low disk storage VM). It will store the .h5 file as well. It will make subfolder inside the log folder with timestamp attached to it.
.h5 files are roughly 200 - 300 MB per epoch. But guess what this log directory is Tensorboard compatible. You can pass the timestamped subfolder as --logdir argument while running tensorboard.
This .py file also has two classes - one class with suffix as Config and another class with suffix as Dataset.
In Config class override the required stuff like
NAME : a name for your project.
NUM_CLASSES : it should be one more than your label class because background is also considered as one label
DETECTION_MIN_CONFIDENCE : by default 0.9 (decrease it if your training images are not of very high quality)
STEPS_PER_EPOCH etc
In Dataset class override the following methods
load_(name_of_the_sample_project) for eg load_balloon
load_mask
image_reference
train function (outside Dataset class) : if you have to change the number of epochs or learning rate etc
All the above mentioned functions are already well-commented so you can follow them to override them for your needs.
