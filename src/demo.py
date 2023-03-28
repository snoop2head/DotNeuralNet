"""
Reference
- https://docs.streamlit.io/library/api-reference/layout
- https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/app.py
- https://huggingface.co/keremberke/yolov8m-valorant-detection/tree/main
- https://docs.ultralytics.com/usage/python/
"""
import time
import PIL

import streamlit as st
import torch
from ultralyticsplus import YOLO, render_result

from convert import convert_to_braille_unicode, parse_xywh_and_class


def load_model(model_path):
    """load model from path"""
    model = YOLO(model_path)
    return model


def load_image(image_path):
    """load image from path"""
    image = PIL.Image.open(image_path)
    return image


# title
st.title("Braille Pattern Detection")

# sidebar
st.sidebar.header("Detection Config")

conf = float(st.sidebar.slider("Class Confidence", 10, 75, 15)) / 100
iou = float(st.sidebar.slider("IoU Threshold", 10, 75, 15)) / 100

model_path = "snoop2head/yolov8m-braille"

try:
    model = load_model(model_path)
    model.overrides["conf"] = conf  # NMS confidence threshold
    model.overrides["iou"] = iou  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 1000  # maximum number of detections per image

except Exception as ex:
    print(ex)
    st.write(f"Unable to load model. Check the specified path: {model_path}")

source_img = None

source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
)
col1, col2 = st.columns(2)

# left column of the page body
with col1:
    if source_img is None:
        default_image_path = "./images/alpha-numeric.jpeg"
        image = load_image(default_image_path)
        st.image(
            default_image_path, caption="Example Input Image", use_column_width=True
        )
    else:
        image = load_image(source_img)
        st.image(source_img, caption="Uploaded Image", use_column_width=True)

# right column of the page body
with col2:
    with st.spinner("Wait for it..."):
        start_time = time.time()
    try:
        with torch.no_grad():
            res = model.predict(
                image, save=True, save_txt=True, exist_ok=True, conf=conf
            )
            boxes = res[0].boxes  # first image
            res_plotted = res[0].plot()[:, :, ::-1]

            list_boxes = parse_xywh_and_class(boxes)

            st.image(res_plotted, caption="Detected Image", use_column_width=True)
            IMAGE_DOWNLOAD_PATH = f"runs/detect/predict/image0.jpg"

    except Exception as ex:
        st.write("Please upload image with types of JPG, JPEG, PNG ...")


try:
    st.success(f"Done! Inference time: {time.time() - start_time:.2f} seconds")
    st.subheader("Detected Braille Patterns")
    for box_line in list_boxes:
        str_left_to_right = ""
        box_classes = box_line[:, -1]
        for each_class in box_classes:
            str_left_to_right += convert_to_braille_unicode(
                model.names[int(each_class)]
            )
        st.write(str_left_to_right)
except Exception as ex:
    st.write("Please try again with images with types of JPG, JPEG, PNG ...")

with open(IMAGE_DOWNLOAD_PATH, "rb") as fl:
    st.download_button(
        "Download object-detected image",
        data=fl,
        file_name="image0.jpg",
        mime="image/jpg",
    )
