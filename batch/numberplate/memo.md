# Calpis example
/sample/calpis folder

## modified files
calpis.py
inspect_calpis_data.ipynb

## VIA annotations
/datasets/calpis/train
no rectangles only polygons


## train from sample/calpis/ folder
python calpis.py train --dataset=../../datasets/calpis --weights=coco

python calpis.py train --dataset=../../datasets/calpis --weights=weights.h5


**resume training**
python calpis.py train --dataset=../../datasets/calpis --weights=last




# TensorFlow install introduction

## tensorflow gpu or cpu
pip install --ignore-installed --upgrade tensorflow-gpu==1.8.0

pip install --ignore-installed (--upgrade) tensorflow

## CUDA
9.0 for 1.8.0 tensorflow
9.1 doesnt work sucks

check out the visual studio integration,

## cudnn
7.1
cudnn-9.0-windows10-x64-v7.1
/bin
/include
/lib


## configurations
**config.py**
adjust from 800,1024 to 400,512 to save GPU memory
IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_DIM = 400
IMAGE_MAX_DIM = 512


## detection
修改了inspect model 的jupyter notebook， 读取calpis config，然后读取 calpis下面的val文件夹
指定 h5文件名为 mask_rcnn_calpis water.h5
"BALLON_WEIGHTS_PATH = \"/path/to/mask_rcnn_calpis water.h5\"  # TODO: update this path"

## oom
try to use ResNet50 as BACKBONE 解決
https://github.com/matterport/Mask_RCNN/blob/e67a6c8ae8b93aa5b049f3da9d467d23e22e3f01/samples/nucleus/nucleus.py


## detection 随机图片
实例  ../sample/calpis water detection mask r-cnn.ipynb

需要更改
import
weight path
CALPIS_WEIGHTS_PATH = "../mask_rcnn_calpis water.h5"

IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(calpis.CalpisConfig):

class_names = ['BG', 'calpis']




## visualize.py 隠す
apply_mask
set alpha=1

comment out # Bounding box and # Label in def display_instances


## save image
skimage.io.imsave('image_save.jpg',image)
もとの画像だけ保存される

visualize.py #172
plt.imsave('plt_imsave_img.jpg',masked_image.astype(np.uint8))
function内保存

visualize.py #174
return masked_image.astype(np.uint8)

print_img= visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
skimage.io.imsave('image_save.jpg',print_img)

jupyter notebook内保存
