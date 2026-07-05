"""Fast training-pipeline tests: the loop learns and closes."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def test_segmentation_learns(tmp_path):
    """Foreground Dice should rise well above its near-zero baseline."""
    from training.train import train_model
    s = train_model("ct_chest_nodule_seg", epochs=5, size=(16, 16, 8),
                    n_train=32, n_val=8, out_dir=str(tmp_path),
                    calibration_json=str(tmp_path / "calib.json"))
    assert s["val_metric_after"] > s["val_metric_before"] + 0.15
    assert Path(s["checkpoint"]).exists()


def test_checkpoint_loads_into_registry(tmp_path):
    """Saved weights must load back into a registry-built model."""
    from training.train import train_model
    from model_registry import ModelRegistry
    s = train_model("ct_chest_nodule_seg", epochs=1, size=(16, 16, 8),
                    n_train=16, n_val=8, out_dir=str(tmp_path),
                    calibration_json=None)
    model = ModelRegistry(device="cpu").load_weights("ct_chest_nodule_seg",
                                                     s["checkpoint"])
    assert model is not None


def test_calibration_stored(tmp_path):
    """Training should fit and persist a temperature for the model."""
    from training.train import train_model
    from calibration import CalibrationStore
    cj = tmp_path / "calib.json"
    train_model("cxr_multilabel_classifier", epochs=2, n_train=32, n_val=8,
                out_dir=str(tmp_path), calibration_json=str(cj))
    assert CalibrationStore(str(cj)).get("cxr_multilabel_classifier") != 1.0
