import os
import random
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from utils import get_alphabet_map

default_transform = transforms.Compose(
    [
        transforms.Resize((40, 25)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.RandomAdjustSharpness(p=0.2, sharpness_factor=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.62784992, 0.62404451, 0.60055435], [0.12606049, 0.11872653, 0.12450065]
        ),
    ]
)


class BrailleDataset(Dataset):
    def __init__(self, transform=default_transform):
        self.kaggle_path = "./dataset/KaggleDataset/Braille Dataset"
        self.angelina_path = "./dataset/AngelinaDataset/AngelinaDataset/cropped_images"
        self.dsbi_path = "./dataset/DSBI/DSBI/cropped_images"
        assert os.path.exists(self.kaggle_path)
        assert os.path.exists(self.angelina_path)
        assert os.path.exists(self.dsbi_path)

        self.alphabet_map = get_alphabet_map(path="./src/utils/alphabet_map.json")
        self.transform = transform

        self.kaggle_files = glob(self.kaggle_path + "/*.jpg")
        self.angelina_files = glob(self.angelina_path + "/*.jpg")
        self.dsbi_files = glob(self.dsbi_path + "/*.jpg")
        self.files = self.kaggle_files + self.angelina_files + self.dsbi_files

        self.kaggle_labels = [self.get_kaggle_label(f) for f in self.kaggle_files]
        self.angelina_labels = [self.get_angelina_label(f) for f in self.angelina_files]
        self.dsbi_labels = [self.get_dsbi_label(f) for f in self.dsbi_files]
        self.labels = self.kaggle_labels + self.angelina_labels + self.dsbi_labels

    def __len__(self):
        return len(self.kaggle_files) + len(self.angelina_files) + len(self.dsbi_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # # get random probability
        p_hflip = random.random()
        p_vflip = random.random()

        # vertical flip
        if p_vflip > 0.5:
            image = torch.flip(image, [1])
            # half the label
            half = label[: len(label) // 2]
            another_half = label[len(label) // 2 :]
            label = half[::-1] + another_half[::-1]

        # horizontal flip
        if p_hflip > 0.5:
            image = torch.flip(image, [2])
            # reverse the label
            half = label[: len(label) // 2]
            another_half = label[len(label) // 2 :]
            label = another_half + half

        return image, torch.tensor(label)

    def get_kaggle_label(self, file):
        file = file.split("/")[-1]
        file = file.split(".")[0]
        file = file[0]
        label = self.alphabet_map[file]
        label = [int(c) for c in label]
        return label

    def get_angelina_label(self, file):
        basename = os.path.basename(file)
        basename = basename.split("_")[-1]
        basename = basename.split(".")[0]
        label = [int(c) for c in basename]
        return label

    def get_dsbi_label(self, file):
        basename = os.path.basename(file)
        basename = basename.split("_")[-1]
        basename = basename.split(".")[0]
        label = [int(c) for c in basename]
        return label


class BrailleDataModule(pl.LightningDataModule):
    def __init__(self, dataset=BrailleDataset(), batch_size=256, num_workers=4):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_size = int(0.80 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
