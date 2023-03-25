"""
Reference
- https://docs.streamlit.io/library/api-reference/layout
- https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/app.py
- https://huggingface.co/keremberke/yolov8m-valorant-detection/tree/main
- https://docs.ultralytics.com/usage/python/
"""
from pathlib import Path
import PIL

import streamlit as st
import torch
from ultralyticsplus import YOLO, render_result

from convert import convert_to_braille_unicode

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
c = st.container()

# left column of the page body

if source_img is None:
    default_image_path = "./images/example.jpeg"
    image = load_image(default_image_path)
    st.image(default_image_path, caption="Example Input Image", use_column_width=True)
else:
    image = load_image(source_img)
    st.image(source_img, caption="Uploaded Image", use_column_width=True)

# right column of the page body

if source_img is None:
    default_detected_image_path = "./images/example_detected.jpeg"
    image = load_image(default_detected_image_path)
    st.image(
        default_detected_image_path,
        caption="Example Detected Image",
        use_column_width=True,
    )
else:
    with torch.no_grad():
        res = model.predict(
            image, save=True, save_txt=True, exist_ok=True, conf=conf
        )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="Detected Image", use_column_width=True)
        IMAGE_DOWNLOAD_PATH = f"runs/detect/predict/image0.jpg"
        with open(IMAGE_DOWNLOAD_PATH, "rb") as fl:
            st.download_button(
                "Download object-detected image",
                data=fl,
                file_name="image0.jpg",
                mime="image/jpg",
            )
        # for r in res:
        #     for c in r.boxes.cls:
        #         print(convert_to_braille_unicode(model.names[int(c)]))
    try:
        with st.expander("Detection Results"):
            for box in boxes:
                st.write(box.xywh)

    except Exception as ex:
        st.write("Please upload image with types of JPG, JPEG, PNG ...")
