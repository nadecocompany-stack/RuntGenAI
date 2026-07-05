"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic runs."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def ct_series(tmp_path):
    """A small synthetic CT chest series directory."""
    from synthetic_dicom import make_series
    return make_series(tmp_path / "ct", "CT", "CHEST", volumetric=True)


@pytest.fixture
def us_image(tmp_path):
    """A single synthetic 2D ultrasound image directory."""
    from synthetic_dicom import make_series
    return make_series(tmp_path / "us", "US", "BREAST", volumetric=False,
                       shape=(128, 128))


@pytest.fixture
def small_unet():
    from monai.networks.nets import UNet
    return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                channels=(16, 32, 64, 128), strides=(2, 2, 2),
                num_res_units=2).eval()
