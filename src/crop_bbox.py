import os
import json
from glob import glob
from PIL import Image

import pandas as pd
import numpy as np
import cv2

from utils import read_DSBI_annotation, transform_angelina_label


def label_to_one_hot(label, num_classes=6):
    one_hot = np.zeros(num_classes)
    for i in range(len(label)):
        one_hot[int(label[i]) - 1] = 1
    # join as str
    one_hot = "".join([str(int(i)) for i in one_hot])
    return one_hot


def crop_dsbi_bbox():
    target_dir="./dataset/DSBI/DSBI/cropped_images"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir,exist_ok=True)
    path = os.getcwd()
    dataset_dir = os.path.join(path, "dataset", "DSBI", "DSBI","data")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Directory '{dataset_dir}' does not exist.")
    images_paths = glob(os.path.join(dataset_dir, "**/*recto.jpg"), recursive=True)
    labels_paths = glob(os.path.join(dataset_dir, "**/*recto.txt"), recursive=True)
    print("Image paths:", images_paths)
    print("Label paths:", labels_paths)
    

    for i, (image_path) in enumerate(images_paths):
        label_path = image_path.replace(".jpg", ".txt")
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        width, height = image.shape[1], image.shape[0]
        list_rects = read_DSBI_annotation(label_path, width, height, 0.3, False)

        for rect in list_rects:
            left, top, right, bottom, label = rect
            cropped_image = image[int(top) : int(bottom), int(left) : int(right)]
            im = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            cropped_image_name = image_name.replace(".jpg", f"_{label}.jpg")
            im.save(os.path.join(target_dir, cropped_image_name))
    print("complete")

def crop_angelina_bbox():
    """crop bounding box and save as cropped images with label name"""

    target_dir="./dataset/AngelinaDataset/AngelinaDataset/cropped_images"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir,exist_ok=True)
    path = os.getcwd()
    dataset_dir = os.path.join(path, "dataset", "AngelinaDataset", "AngelinaDataset")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Directory '{dataset_dir}' does not exist.")
    images_paths = glob(os.path.join(dataset_dir, "**/**/*.jpg"), recursive=True)
    print("Image paths:", images_paths)
    

    for i, (img_path) in enumerate(images_paths):
        # read image
        img = cv2.imread(img_path)
        bbox_path = img_path.replace(".jpg", ".json")
        # read bbox
        with open(bbox_path, "r") as f:
            bbox = json.load(f)
        # Add the validation that removes the pics and src folder
        # crop and save
        for shape in bbox["shapes"]:
            points = np.array(shape["points"])
            x1 = int(points[:, 0].min())
            y1 = int(points[:, 1].min())
            x2 = int(points[:, 0].max())
            y2 = int(points[:, 1].max())
            # crop
            crop_img = img[y1:y2, x1:x2]
            # label
            label = transform_angelina_label(shape["label"])
            label = label_to_one_hot(label)

            # make dir for output
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            # img_path basename + coordinate + label
            save_path = os.path.join(
                target_dir,
                os.path.basename(img_path).replace(".jpg", "")
                + "_"
                + str(x1)
                + "_"
                + str(y1)
                + "_"
                + str(x2)
                + "_"
                + str(y2)
                + "_"
                + label
                + ".jpg",
            )
            try:
                cv2.imwrite(save_path, crop_img)
            except:
                print("error: ", save_path)
                continue
if __name__ == "__main__":
    crop_dsbi_bbox()
    crop_angelina_bbox()