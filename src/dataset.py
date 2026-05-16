"""
Dataset and transforms for brain tumor MRI classification.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class BrainTumorDataset(Dataset):
    """
    Dataset for brain tumor MRI images.

    Expects a DataFrame with at least:
      - 'processed_path': path to the preprocessed PNG image
      - 'label': one of the four class name strings
    """

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["processed_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = CLASS_TO_IDX[row["label"]]
        return image, label


def get_train_transform():
    """Training augmentation pipeline. No horizontal flip (brain asymmetry)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    """Evaluation transform — deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
