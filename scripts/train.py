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

print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False # As a last resort test

def main():
    # UPDATE ME!
    # Configure paths
    user = "jrhowell"
    dataset_dir = f"/home/{user}/sadow_koastore/shared/coral_seg/data/"
    results_dir = f"/home/{user}/benthic_ecology_group/Jack/coral_seg/results/"
    
    #change for the date
    checkpoint_dir = f"{results_dir}/checkpoints_2.13.2026/"

    # Set to None for fresh training
    # Or set to your .ckpt path to load from checkpoint
    #checkpoint_path = f"{results_dir}/add check point/"
    checkpoint_path = None
    
    resume_training = True

    # Configure hyperparameters
    batch_size = 16 
    epochs = 1000
    split_ratio = 0.8
    num_workers = 4
    learning_rate = 3e-4
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
        print(example_batch.keys() if isinstance(example_batch, dict) else type(example_batch))

    # Initialize the model
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = CoralSegFormer.load_from_checkpoint(
            checkpoint_path,
            learning_rate=learning_rate
        )
    else:
        print("Initializing model from scratch")
        model = CoralSegFormer(learning_rate=learning_rate)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= checkpoint_dir,
        filename='coral-segformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        save_last=True,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
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
        # Train
    print("Starting Training...")

    if checkpoint_path is not None and Path(checkpoint_path).exists() and resume_training:
        print("Resuming full training state from checkpoint...")
        trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        print("Starting new training run...")
        trainer.fit(model, data_module)

    print(f"Completed training. Files can be found in {checkpoint_dir}")

if __name__ == "__main__":
    main()