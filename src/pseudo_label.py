import os
import shutil
from glob import glob
from PIL import Image

import torch
from torchvision import transforms

from utils import get_label, get_image, crop_image
from model import BrailleTagger

if __name__ == "__main__":
    default_transform = transforms.Compose(
        [
            transforms.Resize((40, 25)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.61567986, 0.6217983, 0.60718144],
                [0.13666723, 0.13178031, 0.13231377],
            ),
        ]
    )

    model = BrailleTagger()
    ckpt_path = "/root/connect_the_dots/epoch=21-step=5912.ckpt"  # data augmentation + proper resizing + proper normalization
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.eval()

    images_paths = glob("/root/connect_the_dots/YOLODataset/images/**/*.jpg")
    labels_paths = [
        path.replace("images", "labels").replace("jpg", "txt") for path in images_paths
    ]
    target_label_paths = "/root/connect_the_dots/YOLODataset/fixed_labels"
    len(images_paths), len(labels_paths)

    for i, (image_path, label_path) in enumerate(zip(images_paths, labels_paths)):
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        dir_name = os.path.basename(os.path.dirname(image_path))
        if not os.path.exists(os.path.join(target_label_paths, dir_name)):
            os.makedirs(os.path.join(target_label_paths, dir_name))
        label = get_label(label_path)
        if len(label) == 0:
            shutil.copy(
                label_path, os.path.join(target_label_paths, dir_name, label_name)
            )
            print("No boxes, continue")
            continue
        img = get_image(image_path)
        cropped_images = crop_image(img, label)

        print(
            f"Processing {i}th image, {image_name} / {label_name} in {dir_name} directory with {len(cropped_images)} bbox images"
        )

        batch = torch.Tensor()
        for i, cropped_img in enumerate(cropped_images):
            cropped_img = Image.fromarray(cropped_img)
            tensor_img = default_transform(cropped_img)
            tensor_img = tensor_img.unsqueeze(0)
            batch = torch.cat((batch, tensor_img), dim=0)
        output = model(batch)
        output = output.detach().numpy()
        output = output > 0.5
        output = output * 1
        output = output.tolist()
        str_outputs = ["".join([str(item) for item in line]) for line in output]
        new_label = []
        for i, line in enumerate(label):
            new_label.append([str_outputs[i]] + line)
        new_label = [
            line[0] + " " + " ".join([str(item) for item in line[1:]])
            for line in new_label
        ]
        # write the new label as txt file
        with open(os.path.join(target_label_paths, dir_name, label_name), "w") as f:
            f.write("\n".join(new_label))
