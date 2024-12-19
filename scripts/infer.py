import os
from PIL import Image
import numpy as np
import cv2
hot_anno_path = "/home/ubuntu/wyx/program/datasets/HOT/HOT-Annotated"
image_path = os.path.join(hot_anno_path, "images")
segment_path = os.path.join(hot_anno_path, "segments")
temp_path = os.path.join(hot_anno_path, "temp")

if not os.path.exists(temp_path):
    os.mkdir(temp_path)

# convert image
img_name_list = os.listdir(image_path)
print(len(img_name_list))
print(img_name_list[:10])
count_ = len(img_name_list)
for i, filename in enumerate(img_name_list):
    img_path = os.path.join(image_path, filename)
    filename = os.path.basename(img_path)
    print(count_)
    count_ -= 1

    if not img_path.endswith(".png"):
       os.system("convert {} {}".format(img_path, os.path.join(temp_path, ".".join(filename.split(".")[:-1]) + ".png"))) 
    else:
        os.system("cp {} {}".format(img_path, os.path.join(temp_path, filename)))

# mask
mask_name_list = os.listdir(segment_path)
print(len(mask_name_list))
print(mask_name_list[:10])
count_ = len(mask_name_list)
for i, filename in enumerate(mask_name_list):
    mask_path = os.path.join(segment_path, filename)
    filename = os.path.basename(mask_path)
    print(count_)
    count_ -= 1
    img_mask = Image.open(mask_path)
    img_mask = np.array(img_mask)
    img_mask[img_mask > 0] = 255
    kernel = np.ones((60, 60))
    img_mask = cv2.dilate(img_mask, kernel)
    img_mask_save = Image.fromarray(img_mask)
    img_mask_save.save(os.path.join(temp_path, ".".join(filename.split(".")[:-1]) + "_mask001.png"))

    print(mask_path)
    print(filename)

os.system("CUDA_VISIBLE_DIVICES=3 python bin/predict.py model.path=$(pwd)/big-lama indir=/home/ubuntu/wyx/program/datasets/HOT/HOT-Annotated/temp outdir=/home/ubuntu/wyx/program/datasets/HOT/HOT-Annotated/inpainting")