# Calpis example
/sample/calpis folder

## modified files
calpis.py
inspect_calpis_data.ipynb

## VIA annotations
/datasets/calpis/train
no rectangles only polygons


## train from sample/calpis/ folder
need to save mask_rcnn_coco.h5 in ROOT_DIR

python numberplate.py train --dataset=../../datasets/numberplate --weights=coco

python numberplate.py train --dataset=../../datasets/numberplate --weights=last

python calpis.py train --dataset=../../datasets/calpis --weights=coco

python {name_of_the_sample_project}.py train --dataset=../../datasets/{name_of_the_sample_project} --weights=coco or imagenet


**resume training**
python calpis.py train --dataset=../../datasets/calpis --weights=last


## train oom问题
修改config backbone=resnet50
然后 pip install --ignore-installed --upgrade tensorflow-gpu==1.8.0 重新安装
或者需要减少训练图片的数量 大概50张左右
并减少 EPOCH steps =50
即使oom，也可以训练完之后 weights=last 继续训练
修改config
IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_DIM = 512
IMAGE_MAX_DIM = 512
训练的图片里面尽量少物体，会影响到 TRAIN_ROIS_PER_IMAGE = 200

有时候重启pc会改善


https://www.bountysource.com/issues/57710509-running-out-of-memory-while-training

@jameschartouni
Hi!
I also got this error before.
But when I reinstalled everything, it was solved, and I've got some detection results successfully after training on resnet101. I don't know which one works:
Cuda V 9.0.176
cudnn V 7.0.5
NVIDIA driver: 384.111
build a virtual environment(not built via anaconda) to install tensorflow-gpu=1.5.0


BATCH_SIZE=GPU_COUNT * IMAGES_PER_GPU=1

model.py
max_queue_size=100, を減らしてみる  50 にしたら解決


################# OOM ############################
20180519 OOM解決した環境は
Cuda V 9.0.176
cudnn V 7.0.5
tensorflow-gpu 1.8.0

BATCH_SIZE=GPU_COUNT * IMAGES_PER_GPU=1
model.py
max_queue_size=50

EPOCH steps =100
#############################################




# TensorFlow install introduction

## tensorflow gpu or cpu
pip install --ignore-installed --upgrade tensorflow-gpu==1.8.0

pip install --ignore-installed (--upgrade) tensorflow

## CUDA
9.0 for 1.8.0 tensorflow

9.1 doesnt work sucks

check out the visual studio integration,

## cudnn
cudnn-9.0-windows10-x64-v7.zip

7.1
cudnn-9.0-windows10-x64-v7.1
/bin
/include
/lib


## best env
Cuda V 9.0.176
cudnn V 7.0.5
NVIDIA driver: 384.111
build a virtual environment(not built via anaconda) to install tensorflow-gpu=1.8.0


## configurations
**config.py**
adjust from 800,1024 to 400,512 to save GPU memory
IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_DIM = 512
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

## train 精度
IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_DIM = 512
IMAGE_MAX_DIM = 640
min max を調整することで精度に影響

640にしたら、半分隠されている小さいプレートも検知できる
だけど車体に移っている小さい看板も誤認識してしまう

512にしたら、半分隠されている小さいプレートが検知できなくなる
np_train (50).jpgの右

512はトレーニングするときの設定値だったので、一番適性がいい

## LOGGING処理
面倒くさいから、printをそのままlogファイルに書き込む
https://stackoverflow.com/questions/2513479/redirect-prints-to-log-file

もしくは正しいやり方で
import logging
https://docs.python.jp/3/howto/logging.html

例：
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("hello")

Produce a file named "logfile" with content:

2012-10-18 06:40:03,582 INFO     hello
