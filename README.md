# DotNeuralNet

[![Streamlit Demo App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/snoop2head/braille-detection)

**Light-weight Neural Network for Optical Braille Recognition in the wild & on the book.**

- Classified multi label one-hot encoded labels for raised dots.
- Pseudo-labeled Natural Scene Braille symbols.
- Trained single stage object detection YOLO models for Braille symbols.

### Repository Structure

```
DotNeuralNet
ㄴ assets - example images and train/val logs
ㄴ dataset
  ㄴ AngelinaDataset - book background
  ㄴ braille_natural - natural scene background
  ㄴ DSBI - book background
  ㄴ KaggleDataset - arbitrary 6 dots
  ㄴ yolo.yaml - yolo dataset config
ㄴ src
  ㄴ utils
    ㄴ angelina_utils.py
    ㄴ braille_natural_utils.py
    ㄴ dsbi_utils.py
    ㄴ kaggle_utils.py
  ㄴ crop_bbox.py
  ㄴ dataset.py
  ㄴ model.py
  ㄴ pseudo_label.py
  ㄴ train.py
  ㄴ visualize.py
ㄴ weights
  ㄴ yolov5_braille.pt # yolov5-m checkpoint
  ㄴ yolov8_braille.pt # yolov8-m checkpoint
```

### Result

- Inferenced result of yolov8-m model on validation subset.
  ![yolov8 img](./assets/result_yolov8.png)
- Inferenced result of yolov5-m model on validation subset.
  ![yolov5 img](./assets/result_yolov5.png)

### Logs

- Train / Validation log of yolov8-m model
  ![yolov8 log](./assets/log_yolov8_long.png)
- Train / Validation log of yolov5-m model available at [🔗 WandB](https://wandb.ai/snoop2head/YOLOv5/runs/mqvmh4nc)
  ![yolov8 log](./assets/log_yolov5.png)

### Installation

CV2 and Yolo Dependency Installation

```shell
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

### How to Run

- Please refer to `src/inference.py` or `src/demo.py` to run the model.
- For online demo, please visit [🔗 Streamlit demo](https://huggingface.co/spaces/snoop2head/braille-detection).

```python
import PIL
from ultralytics import YOLO
from convert import convert_to_braille_unicode, parse_xywh_and_class

def load_model(model_path):
    """load model from path"""
    model = YOLO(model_path)
    return model

def load_image(image_path):
    """load image from path"""
    image = PIL.Image.open(image_path)
    return image

# constants
CONF = 0.15 # or other desirable confidence threshold level
MODEL_PATH = "./weights/yolov8_braille.pt"
IMAGE_PATH = "./assets/alpha-numeric.jpeg"

# receiving results from the model
image = load_image(IMAGE_PATH)
model = YOLO(MODEL_PATH)
res = model.predict(image, save=True, save_txt=True, exist_ok=True, conf=CONF)
boxes = res[0].boxes  # first image
list_boxes = parse_xywh_and_class(boxes)

result = ""
for box_line in list_boxes:
    str_left_to_right = ""
    box_classes = box_line[:, -1]
    for each_class in box_classes:
        str_left_to_right += convert_to_braille_unicode(model.names[int(each_class)])
    result += str_left_to_right + "\n"

print(result)
"""
⠁⠃⠉⠋⠙⠑⠙⠋⠛⠓⠊⠑
⠓⠇⠇⠍⠝⠕⠏⠟⠗
⠎⠞⠥⠼⠗⠭⠵
⠼⠧⠚⠁⠃⠉⠙⠑⠙⠛⠚⠊⠑
"""

```

### Citation

If you find DotNeuralNet useful for your research, please consider citing the repository:

```
@misc{ahn2024dotneuralnet,
  author={Ahn, Young Jin},
  title={DotNeuralNet: Light-weight Neural Network for Optical Braille Recognition in the Wild},
  year={2023},
}
```