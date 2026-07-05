"""
datamodule.py
==================================================================
LightningDataModule wrapping the synthetic dataset, plus the contract a real
dataset adapter must satisfy and a per-dataset requirements table (the answer
to "what data do you need?").

Swap synthetic for real by implementing a Dataset that yields the SAME batch
dict — {"image": (C,*spatial) float in [0,1], "label": ...} — where label is:
  * segmentation, multiclass : (1, *spatial) int64
  * segmentation, multilabel : (C, *spatial) float {0,1}
  * classification, multiclass : scalar int64
  * classification, multilabel : (C,) float {0,1}
...then pass it to RadDataModule(train_ds=..., val_ds=...).
"""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from label_taxonomies import ModelSpec
from training.synthetic_dataset import SyntheticScanDataset


# What each real dataset requires — obtain, then write an adapter Dataset.
DATA_REQUIREMENTS = {
    "ct_chest_nodule_seg": {
        "dataset": "LIDC-IDRI",
        "source": "TCIA (registration required)",
        "format": "DICOM CT series + XML nodule contours",
        "adapter_notes": "Build consensus nodule masks (e.g. pylidc, >=50% "
                         "reader agreement); yield (1,H,W,D) volume + (1,H,W,D) mask.",
    },
    "ct_head_ich_classifier": {
        "dataset": "RSNA Intracranial Hemorrhage Detection",
        "source": "Kaggle competition data",
        "format": "DICOM slices + per-slice CSV labels (5 subtypes + 'any')",
        "adapter_notes": "3-window slice stacks as channels; yield image + (6,) "
                         "multilabel float. No masks (localization via Grad-CAM).",
    },
    "mr_brain_tumor_seg": {
        "dataset": "BraTS",
        "source": "Synapse / challenge registration",
        "format": "4 co-registered NIfTI sequences (T1,T1ce,T2,FLAIR) + label NIfTI",
        "adapter_notes": "Stack 4 sequences as channels; map raw labels to the "
                         "3 nested regions (WT/TC/ET). Confirm the release's "
                         "enhancing-tumor integer (4 vs 3).",
    },
    "us_breast_lesion": {
        "dataset": "BUSI",
        "source": "public download",
        "format": "PNG images + PNG lesion masks, 3 categories",
        "adapter_notes": "yield (1,H,W) image + (3,H,W) or (1,H,W) mask; normal "
                         "images carry an empty mask.",
    },
    "cxr_multilabel_classifier": {
        "dataset": "CheXpert / MIMIC-CXR",
        "source": "Stanford AIMI / PhysioNet (credentialed)",
        "format": "JPG/PNG + CSV with 14 observations (uncertainty labels)",
        "adapter_notes": "Resolve uncertainty labels (U-Ones/U-Zeros/ignore); "
                         "yield RRAD-DINO-preprocessed image + (14,) multilabel float.",
    },
}


class RadDataModule(pl.LightningDataModule):
    """
    Data module. With no real datasets supplied it builds synthetic data from
    the model spec; otherwise it uses the datasets you pass in.
    """

    def __init__(self, spec: ModelSpec, batch_size: int = 4, num_workers: int = 0,
                 n_train: int = 64, n_val: int = 16, size=None,
                 train_ds: Optional[Dataset] = None, val_ds: Optional[Dataset] = None):
        super().__init__()
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train = train_ds
        self._val = val_ds
        self._n_train, self._n_val, self._size = n_train, n_val, size

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train is None:
            self._train = SyntheticScanDataset(self.spec, n=self._n_train,
                                               train=True, size=self._size)
        if self._val is None:
            self._val = SyntheticScanDataset(self.spec, n=self._n_val,
                                             train=False, size=self._size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
