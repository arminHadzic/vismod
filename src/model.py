import timm

import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class ModClassifier(pl.LightningModule):

  def __init__(self,
               model_name: str = "convnextv2_tiny.fcmae_ft_in1k",
               lr: float = 1e-4):
    super().__init__()
    self.save_hyperparameters()

    self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1,            # binary classification
            global_pool="avg",        # ensures output shape is (B, C)
        )

    self.acc = BinaryAccuracy()
    self.auc = BinaryAUROC()
    self.f1 = BinaryF1Score()

  def forward(self, x):
    return self.model(x)

  def shared_step(self, batch, stage):
    x, y = batch
    logits = self(x).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, y.float())

    preds = torch.sigmoid(logits)
    self.log(f"{stage}_loss", loss, prog_bar=True)
    self.log(f"{stage}_acc", self.acc(preds, y), prog_bar=True)
    self.log(f"{stage}_f1", self.f1(preds, y))
    self.log(f"{stage}_auc", self.auc(preds, y))
    return loss

  def training_step(self, batch, batch_idx):
    return self.shared_step(batch, "train")

  def validation_step(self, batch, batch_idx):
    self.shared_step(batch, "val")

  def test_step(self, batch, batch_idx):
    self.shared_step(batch, "test")

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
