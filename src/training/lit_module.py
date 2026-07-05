"""
lit_module.py
==================================================================
A single LightningModule that trains any of the platform's models, dispatching
loss and metric by task type read from the model spec:

  segmentation (multiclass)  -> DiceCELoss(softmax) ; metric = foreground Dice
  segmentation (multilabel)  -> DiceCELoss(sigmoid) ; metric = foreground Dice
  classification (multiclass)-> CrossEntropy        ; metric = accuracy
  classification (multilabel)-> BCEWithLogits       ; metric = mean acc@0.5

Metrics are computed manually (no torchmetrics dependency) to keep the stack
small. After fit, train.py runs a validation pass to fit the calibration
temperature and store it.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn

from monai.losses import DiceCELoss

from label_taxonomies import ModelSpec, Localization


class RadLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, spec: ModelSpec, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.spec = spec
        self.lr = lr
        self.is_seg = spec.localization is Localization.MASK_CC
        self.multilabel = spec.multilabel

        if self.is_seg:
            self.loss_fn = (DiceCELoss(sigmoid=True) if self.multilabel
                            else DiceCELoss(to_onehot_y=True, softmax=True))
        else:
            self.loss_fn = (nn.BCEWithLogitsLoss() if self.multilabel
                            else nn.CrossEntropyLoss())
        self._val = []      # (metric_sum, count) accumulation per epoch

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, y):
        return self.loss_fn(logits, y)

    def training_step(self, batch, _):
        logits = self.model(batch["image"])
        loss = self._loss(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_start(self):
        self._val = []

    def validation_step(self, batch, _):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self._loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self._val.append(self._metric(logits.detach(), y))

    def on_validation_epoch_end(self):
        if self._val:
            self.log("val_metric", float(sum(self._val) / len(self._val)),
                     prog_bar=True)

    # -- task-specific metric ---------------------------------------------
    def _metric(self, logits, y) -> float:
        if self.is_seg:
            if self.multilabel:
                pred_fg = (torch.sigmoid(logits) > 0.5).any(dim=1)
                gt_fg = (y > 0.5).any(dim=1)
            else:
                pred_fg = logits.argmax(dim=1) > 0
                gt_fg = y[:, 0] > 0
            inter = (pred_fg & gt_fg).sum().float()
            denom = pred_fg.sum().float() + gt_fg.sum().float()
            return float((2 * inter / denom).item()) if denom > 0 else 1.0
        # classification
        if self.multilabel:
            pred = (torch.sigmoid(logits) > 0.5).float()
            return float((pred == y).float().mean().item())
        return float((logits.argmax(dim=1) == y).float().mean().item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
