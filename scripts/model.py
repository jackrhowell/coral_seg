import torch
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from torch.optim import AdamW

class CoralSegFormer(pl.LightningModule):
    def __init__(self, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Define new labels
        self.id2label = {
            0: "background",
            1: "nv",
            2: "nh",
            3: "hy",
            4: "st-zo"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Load Pretrained Model
        # We use ignore_mismatched_sizes=True because we are changing 
        # the number of labels from the ImageNet default to 5.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", # Source checkpoint
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        # SegFormer calculates loss internally if labels are provided
        outputs = self(images, masks)
        loss = outputs.loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images, masks)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Using AdamW. Source suggested 3e-3, but 3e-4 is often safer 
        # for stability when fine-tuning a custom dataset.
        return AdamW(self.model.parameters(), lr=self.learning_rate)