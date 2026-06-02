import torch
import pytorch_lightning as pl
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class CoralSegFormer(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()

        self.lr = lr 

        self.id2label = {
            0: "background",
            1: "nv",
            2: "nh",
            3: "hy",
            4: "st_zo"
        }

        self.label2id = {v: k for k, v in self.id2label.items()}

        config = SegformerConfig.from_pretrained(
            "nvidia/mit-b0",
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,

            # Dropout values
            hidden_dropout_prob=0.85,
            attention_probs_dropout_prob=0.85
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            config=config,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(pixel_values=images, labels=masks)
        loss = outputs.loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(pixel_values=images, labels=masks)
        loss = outputs.loss
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-2
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }