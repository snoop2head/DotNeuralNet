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


def crop_dsbi_bbox(target_dir="./cropped_images/braille_classification_DSBI"):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    path = os.getcwd()
    images_paths = glob(os.path.join(path, "../dataset/DSBI/**/*recto.jpg"))
    labels_paths = glob(os.path.join(path, "../dataset/DSBI/**/*recto.txt"))

    os.makedirs(target_dir, exist_ok=True)

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


def crop_angelina_bbox(
    img_path, bbox_path, target_dir="./cropped_images/braille_classification_angelina"
):
    """crop bounding box and save as cropped images with label name"""
    # read image
    img = cv2.imread(img_path)
    # read bbox
    with open(bbox_path, "r") as f:
        bbox = json.load(f)
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
