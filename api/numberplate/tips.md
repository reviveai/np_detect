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
