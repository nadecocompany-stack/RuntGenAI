"""
synthetic_dataset.py
==================================================================
Synthetic *labeled* data for exercising the training pipeline end to end. The
signal is deliberately learnable so a short run demonstrably reduces loss and
improves the metric — proving the loop (data → loss → backprop → optimizer →
metric → calibration) is wired correctly, not just that it executes.

Two generators, matched to the two task types:
  * segmentation  — a bright blob whose voxels are the foreground mask.
  * classification — class-specific TEXTURE FREQUENCIES added to the image, so
    the label survives global pooling (a frequency-selective conv filter can
    detect each class). Multilabel superimposes several; multiclass adds one.

This is NOT medical data. Real datasets plug in via datamodule.py adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from label_taxonomies import ModelSpec, Localization
from model_registry import _spatial_dims, _in_channels


@dataclass
class TaskConfig:
    is_seg: bool
    multilabel: bool
    num_classes: int
    spatial_dims: int
    in_channels: int
    size: Tuple[int, ...]

    @classmethod
    def from_spec(cls, spec: ModelSpec, size: Optional[Tuple[int, ...]] = None):
        sd = _spatial_dims(spec)
        if size is None:
            # divisible by the U-Net stride product (2^3=8); 16 keeps margin
            size = (32, 32, 16) if sd == 3 else (64, 64)
        return cls(
            is_seg=(spec.localization is Localization.MASK_CC),
            multilabel=spec.multilabel,
            num_classes=spec.num_classes,
            spatial_dims=sd,
            in_channels=_in_channels(spec),
            size=tuple(size),
        )


def _sphere_mask(size, center, radius) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, s) for s in size)]
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return dist2 <= radius ** 2


def _texture(size, freq, phase=0.0) -> np.ndarray:
    """A separable sinusoidal texture of a given spatial frequency."""
    axes = [np.cos(2 * np.pi * freq * (np.arange(s) / max(s, 1)) + phase) for s in size]
    out = axes[0]
    for a in axes[1:]:
        out = np.add.outer(out, a)
    return out.reshape(size)


class SyntheticScanDataset(Dataset):
    """Deterministic per-index synthetic samples for one model spec."""

    def __init__(self, spec: ModelSpec, n: int = 64, train: bool = True,
                 size: Optional[Tuple[int, ...]] = None, seed: int = 0,
                 signal: float = 0.9) -> None:
        self.cfg = TaskConfig.from_spec(spec, size)
        self.n = n
        self.signal = signal
        self.base_seed = seed + (0 if train else 10_000)
        # distinct frequency per class for the classification texture code
        self._freqs = np.linspace(2.0, 8.0, max(self.cfg.num_classes, 2))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)
        c = self.cfg
        if c.is_seg:
            return self._make_seg(rng)
        return self._make_cls(rng)

    # -- segmentation ------------------------------------------------------
    def _make_seg(self, rng):
        c = self.cfg
        img = rng.normal(0.2, 0.05, (c.in_channels, *c.size)).astype(np.float32)
        center = [rng.integers(s // 4, 3 * s // 4) for s in c.size]
        radius = min(c.size) * 0.2
        sph = _sphere_mask(c.size, center, radius)
        cls = 1 if c.num_classes <= 2 else int(rng.integers(1, c.num_classes))
        img[:, sph] += self.signal

        if c.multilabel:                          # (C, *spatial) float one-hot
            label = np.zeros((c.num_classes, *c.size), np.float32)
            label[cls][sph] = 1.0
        else:                                     # (1, *spatial) integer
            label = np.zeros((1, *c.size), np.int64)
            label[0][sph] = cls
        return {"image": torch.from_numpy(np.clip(img, 0, 1)),
                "label": torch.from_numpy(label)}

    # -- classification ----------------------------------------------------
    def _make_cls(self, rng):
        c = self.cfg
        img = rng.normal(0.3, 0.05, (c.in_channels, *c.size)).astype(np.float32)
        if c.multilabel:
            y = (rng.random(c.num_classes) < 0.4).astype(np.float32)
            for k in np.nonzero(y)[0]:
                img += self.signal * _texture(c.size, self._freqs[k],
                                              phase=rng.random()) / 3.0
            label = torch.from_numpy(y)           # (C,) float
        else:
            cls = int(rng.integers(0, c.num_classes))
            img += self.signal * _texture(c.size, self._freqs[cls])
            label = torch.tensor(cls, dtype=torch.long)
        return {"image": torch.from_numpy(np.clip(img, 0, 1).astype(np.float32)),
                "label": label}
