"""
train.py
==================================================================
Train one model (by registry key), then close the loop:
  1. Lightning fit on the data module (synthetic unless real datasets supplied).
  2. Validation pass to collect logits/labels.
  3. Fit the calibration temperature on those and store it (calibration.py).
  4. Save a plain state_dict that ModelRegistry.load_weights can consume.

Run:  python -m training.train ct_chest_nodule_seg --epochs 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch

from label_taxonomies import get_spec, Localization
from model_registry import ModelRegistry
from calibration import fit_temperature, expected_calibration_error, CalibrationStore
from training.datamodule import RadDataModule
from training.lit_module import RadLitModule

logger = logging.getLogger("train")


def _collect_val(module: RadLitModule, dm: RadDataModule):
    """Run the val set, returning flattened (logits, labels) for calibration."""
    module.eval()
    logits_all, labels_all = [], []
    seg, multilabel = module.is_seg, module.multilabel
    with torch.no_grad():
        for batch in dm.val_dataloader():
            logits = module.model(batch["image"])
            y = batch["label"]
            if seg:                                   # (B,C,*sp) -> (M,C)
                c = logits.shape[1]
                logits = logits.movedim(1, -1).reshape(-1, c)
                if multilabel:
                    y = y.movedim(1, -1).reshape(-1, c)
                else:
                    y = y[:, 0].reshape(-1)
            logits_all.append(logits)
            labels_all.append(y)
    L = torch.cat(logits_all); Y = torch.cat(labels_all)
    # cap size so LBFGS stays fast
    if L.shape[0] > 50_000:
        idx = torch.randperm(L.shape[0])[:50_000]
        L, Y = L[idx], Y[idx]
    return L, Y


def train_model(
    key: str,
    epochs: int = 8,
    lr: float = 1e-3,
    size=None,
    n_train: int = 64,
    n_val: int = 16,
    out_dir: str = "checkpoints",
    calibration_json: Optional[str] = "calibration.json",
    accelerator: str = "auto",
    enable_progress_bar: bool = False,
    seed: int = 0,
) -> dict:
    spec = get_spec(key)
    if spec is None:
        raise KeyError(f"unknown registry key '{key}'")
    pl.seed_everything(seed, workers=True)

    model = ModelRegistry(device="cpu", seed=seed).build(key)
    module = RadLitModule(model, spec, lr=lr)
    dm = RadDataModule(spec, size=size, n_train=n_train, n_val=n_val)
    dm.setup()

    trainer = pl.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
    )
    # baseline metric before training (for an honest "it learned" delta)
    base = trainer.validate(module, dm.val_dataloader(), verbose=False)[0]
    trainer.fit(module, dm)
    final = trainer.validate(module, dm.val_dataloader(), verbose=False)[0]

    mode = "multilabel" if spec.multilabel else "multiclass"
    L, Y = _collect_val(module, dm)
    ece_before = expected_calibration_error(
        (torch.sigmoid(L) if mode == "multilabel" else torch.softmax(L, -1)), Y, mode)
    T = fit_temperature(L, Y, mode)
    from calibration import apply_temperature
    ece_after = expected_calibration_error(apply_temperature(L, T, mode), Y, mode)

    if calibration_json:
        CalibrationStore(calibration_json).set(key, T)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt = str(Path(out_dir) / f"{key}.pt")
    torch.save(model.state_dict(), ckpt)

    summary = {
        "key": key, "epochs": epochs,
        "val_metric_before": round(base.get("val_metric", 0.0), 4),
        "val_metric_after": round(final.get("val_metric", 0.0), 4),
        "val_loss_after": round(final.get("val_loss", 0.0), 4),
        "temperature": round(T, 4),
        "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4),
        "checkpoint": ckpt,
    }
    logger.info("trained %s: %s", key, summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    s = train_model(args.key, epochs=args.epochs, lr=args.lr,
                    enable_progress_bar=True)
    print("\n".join(f"  {k}: {v}" for k, v in s.items()))
