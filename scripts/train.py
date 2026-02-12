import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import CoralDataModule
from model import CoralSegFormer
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    # UPDATE ME!
    # Configure paths
    user = "jrhowell"
    dataset_dir = f"/home/{user}/sadow_koastore/shared/coral_seg/coral_seg/data/"
    results_dir = f"/home/{user}/sadow_koastore/shared/coral_seg/coral_seg/results/"
    checkpoint_dir = f"{results_dir}/checkpoints_2.12.2026/"

    # Configure hyperparameters
    batch_size = 8 
    epochs = 30
    split_ratio = 0.8
    num_workers = 4
    samples_per_image = 100
    crop_size = (512, 512)

    # Initialize the data module
    data_module = CoralDataModule(
        root_dir=dataset_dir, 
        batch_size=batch_size, 
        split_ratio=split_ratio,
        num_workers=num_workers,
        samples_per_image=samples_per_image,
        crop_size=crop_size
    )

    data_module.setup() 
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Load an example batch to determine input shape
    example_batch = next(iter(train_loader))
    print(example_batch)

    # Initialize the model
    model = CoralSegFormer(learning_rate=3e-4)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= checkpoint_dir,
        filename='coral-segformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    callbacks = [checkpoint_callback, early_stop_callback]

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto", # Auto-detects GPU/CPU
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10
    )

    # Train
    print("Starting Training...")
    trainer.fit(model, data_module)

    print(f"Completed training. Files can be found in {checkpoint_dir}")

if __name__ == "__main__":
    main()