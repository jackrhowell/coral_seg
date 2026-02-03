import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pytorch_lightning as pl
import random

class CoralRandomCropDataset(Dataset):
    def __init__(
        self, 
        file_list, 
        crop_size=(512, 512), 
        samples_per_image=50
    ):
        """
        Args:
            file_list: A list of tuples [(img_path, mask_path), ...].
            samples_per_image: How many random crops to extract from each image file per epoch.
        """
        self.file_list = file_list
        self.crop_h, self.crop_w = crop_size
        self.samples_per_image = samples_per_image
        
        # Normalization (ImageNet stats for SegFormer)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # Total length is number of files * crops we want per file
        return len(self.file_list) * self.samples_per_image

    def __getitem__(self, idx):
        # Determine which image file corresponds to this index
        file_idx = idx // self.samples_per_image
        image_path, mask_path = self.file_list[file_idx]

        # Load images
        # We load inside getitem to avoid holding all large images in RAM
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        img_w, img_h = image.size
        mask_np = np.array(mask)

        # FIND VALID BOUNDS
        # Find coordinates where mask is not background (>0) to sample interesting areas.
        # This fulfills "randomly sample from within the bounds of the segmentation masks"
        valid_y, valid_x = np.where(mask_np > 0)

        # Safety check: if an image is empty (all background), fallback to center crop
        if len(valid_y) == 0:
            center_y, center_x = img_h // 2, img_w // 2
        else:
            rand_idx = random.randint(0, len(valid_y) - 1)
            center_y, center_x = valid_y[rand_idx], valid_x[rand_idx]

        # Calculate crop coordinates (top-left)
        top = center_y - (self.crop_h // 2)
        left = center_x - (self.crop_w // 2)

        # Boundary checks
        top = max(0, min(top, img_h - self.crop_h))
        left = max(0, min(left, img_w - self.crop_w))

        # Perform Crop
        image_crop = image.crop((left, top, left + self.crop_w, top + self.crop_h))
        mask_crop = mask.crop((left, top, left + self.crop_w, top + self.crop_h))

        # Transform and Remap
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
        samples_per_image=100,
        crop_size=(512, 512)
    ):
        """
        Args:
            root_dir: The top-level directory containing subdirectories with data.
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.train_files = []
        self.val_files = []
        self.num_workers = num_workers
        self.samples_per_image = samples_per_image
        self.crop_size = crop_size

    def setup(self, stage=None):
        # 1. DISCOVERY: Walk through directory to find all valid pairs
        all_files = []
        print(f"Scanning {self.root_dir}...")
        
        for root, dirs, files in os.walk(self.root_dir):
            # Check if both required files exist in this directory
            if "image.png" in files and "seg_r10.png" in files:
                img_path = os.path.join(root, "image.png")
                mask_path = os.path.join(root, "seg_r10.png")
                all_files.append((img_path, mask_path))
        
        print(f"Found {len(all_files)} valid image/mask pairs.")

        # 2. SPLIT: Shuffle and split by FILE, not by crop
        # This prevents training and validating on crops from the same image (leakage)
        random.shuffle(all_files)
        split_idx = int(len(all_files) * self.split_ratio)
        
        self.train_files = all_files[:split_idx]
        self.val_files = all_files[split_idx:]
        
        print(f"Training on {len(self.train_files)} images.")
        print(f"Validating on {len(self.val_files)} images.")

        # 3. CREATE DATASETS
        # samples_per_image dictates how much data we generate from each large map
        self.train_ds = CoralRandomCropDataset(
            self.train_files, 
            samples_per_image=self.samples_per_image,
            crop_size=self.crop_size
        ) 
        
        self.val_ds = CoralRandomCropDataset(
            self.val_files, 
            samples_per_image=self.samples_per_image,
            crop_size=self.crop_size
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