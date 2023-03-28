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
