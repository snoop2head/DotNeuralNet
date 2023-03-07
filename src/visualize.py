import cv2
from matplotlib import pyplot as plt
from utils import read_DSBI_annotation


def read_image(path):
    return cv2.imread(path)


def show_image(image, title=""):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(title)
    plt.show()


def show_image_with_rects(image, list_rects, title):
    image = image.copy()
    for rect in list_rects:
        left, top, right, bottom, label = rect
        cv2.rectangle(
            image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2
        )
        cv2.putText(
            image,
            label,
            (int(left), int(top)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    show_image(image, title)


if __name__ == "__main__":
    sample_img_path = "../dataset/DSBI/SYF+3+recto.jpg"
    sample_label_path = sample_img_path.replace(".jpg", ".txt")
    image = read_image(sample_img_path)

    print(image.shape)
    height, width, _ = image.shape
    list_rects = read_DSBI_annotation(sample_label_path, width, height, 0.3, False)
    print(list_rects)

    show_image_with_rects(image, list_rects, "image with annotation")
