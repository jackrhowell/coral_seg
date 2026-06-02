import os
import random
from collections import Counter

import torch
import numpy as np
import pytorch_lightning as pl

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def dominant_class(mask_path):
    """
    Returns the dominant non-background class in a mask.
    If the mask only contains background, returns 0.
    """
    mask = np.array(Image.open(mask_path))

    vals, counts = np.unique(mask, return_counts=True)

    # Ignore background if possible
    non_bg = vals != 0
    if np.any(non_bg):
        vals = vals[non_bg]
        counts = counts[non_bg]

    return int(vals[np.argmax(counts)])


class CoralRandomCropDataset(Dataset):
    def __init__(
        self,
        file_list,
        crop_size=(512, 512),
        samples_per_image=50,
        augment=False
    ):
        self.file_list = file_list
        self.crop_h, self.crop_w = crop_size
        self.samples_per_image = samples_per_image
        self.augment = augment

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.color_jitter = T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03
        )

    def __len__(self):
        return len(self.file_list) * self.samples_per_image

    def apply_augmentation(self, image, mask):
        if random.random() < 0.67:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.67:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.67:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST
            )

        if random.random() < 0.67:
            image = self.color_jitter(image)

        return image, mask

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_image
        image_path, mask_path = self.file_list[file_idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        img_w, img_h = image.size
        mask_np = np.array(mask)

        valid_y, valid_x = np.where(mask_np > 0)

        if len(valid_y) == 0:
            center_y, center_x = img_h // 2, img_w // 2
        else:
            rand_idx = random.randint(0, len(valid_y) - 1)
            center_y, center_x = valid_y[rand_idx], valid_x[rand_idx]

        top = center_y - (self.crop_h // 2)
        left = center_x - (self.crop_w // 2)

        top = max(0, min(top, img_h - self.crop_h))
        left = max(0, min(left, img_w - self.crop_w))

        image_crop = image.crop((left, top, left + self.crop_w, top + self.crop_h))
        mask_crop = mask.crop((left, top, left + self.crop_w, top + self.crop_h))

        if self.augment:
            image_crop, mask_crop = self.apply_augmentation(image_crop, mask_crop)

        image_tensor = self.transform(image_crop)
        mask_tensor = torch.from_numpy(np.array(mask_crop)).long()

        return image_tensor, mask_tensor


class CoralDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size=8,
        split_ratio=0.8,
        num_workers=4,
        samples_per_image=60,
        crop_size=(512, 512)
    ):
        super().__init__()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.samples_per_image = samples_per_image
        self.crop_size = crop_size

        self.train_files = []
        self.val_files = []

    def setup(self, stage=None):
        all_files = []

        print(f"Scanning {self.root_dir}...")

        for root, dirs, files in os.walk(self.root_dir):
            if "image.png" in files and "seg_r10.png" in files:
                img_path = os.path.join(root, "image.png")
                mask_path = os.path.join(root, "seg_r10.png")
                all_files.append((img_path, mask_path))

        print(f"Found {len(all_files)} valid image/mask pairs.")

        if len(all_files) == 0:
            raise RuntimeError(
                f"No image/mask pairs found in {self.root_dir}. "
                "Expected folders containing image.png and seg_r10.png."
            )

        labels = [dominant_class(mask_path) for _, mask_path in all_files]
        label_counts = Counter(labels)

        print("Dominant class counts:")
        for label, count in sorted(label_counts.items()):
            print(f"  class {label}: {count}")

        # sklearn stratify requires every class to have at least 2 examples.
        # Rare singleton classes are forced into train, then the rest are stratified.
        singleton_files = [
            file_pair
            for file_pair, label in zip(all_files, labels)
            if label_counts[label] < 2
        ]

        strat_files = [
            file_pair
            for file_pair, label in zip(all_files, labels)
            if label_counts[label] >= 2
        ]

        strat_labels = [
            label
            for label in labels
            if label_counts[label] >= 2
        ]

        if len(strat_files) > 1 and len(set(strat_labels)) > 1:
            train_files, val_files = train_test_split(
                strat_files,
                train_size=self.split_ratio,
                shuffle=True,
                stratify=strat_labels
            )
        else:
            # Fallback if there are not enough files/classes to stratify.
            train_files, val_files = train_test_split(
                all_files,
                train_size=self.split_ratio,
                shuffle=True
            )
            singleton_files = []

        # Keep rare singleton classes in training so the model can see them.
        self.train_files = list(train_files) + singleton_files
        self.val_files = list(val_files)

        print(f"Train images: {len(self.train_files)}")
        print(f"Val images: {len(self.val_files)}")

        self.train_ds = CoralRandomCropDataset(
            self.train_files,
            samples_per_image=self.samples_per_image,
            crop_size=self.crop_size,
            augment=True
        )

        self.val_ds = CoralRandomCropDataset(
            self.val_files,
            samples_per_image=self.samples_per_image,
            crop_size=self.crop_size,
            augment=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
