from PIL import Image


def get_label(label_path):
    with open(label_path, "r") as f:
        label = f.readlines()
    label = [line.strip().split(" ") for line in label]
    # label is in yolo format, remove class in the first item of the row
    label = [line[1:] for line in label]

    # convert to float
    label = [[float(item) for item in line] for line in label]

    return label


def get_image(img_path):
    img = Image.open(img_path)
    return img
