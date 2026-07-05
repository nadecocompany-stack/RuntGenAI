"""
calibration.py
==================================================================
Confidence calibration via temperature scaling (Guo et al., 2017), plus the
metric that proves it worked (Expected Calibration Error) and a small store so
a fitted temperature per model is actually applied at inference time.

Why this exists
---------------
Raw softmax/sigmoid outputs from deep nets are typically *over-confident*: a
"90%" prediction is right far less than 90% of the time. Temperature scaling
divides the logits by a single scalar T (fit on held-out validation data) to
fix this without changing which class wins. T > 1 softens (fixes
over-confidence); T = 1 is a no-op.

Handles both regimes used in this platform:
  * multiclass (softmax + cross-entropy)   — e.g. BUSI, nodule seg
  * multilabel (sigmoid + BCE-with-logits) — RSNA-ICH, CheXpert, BraTS regions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("calibration")
logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------
# flatten helpers — collapse any spatial dims into the sample dim
# --------------------------------------------------------------------------
def _flatten(logits: torch.Tensor, labels: torch.Tensor, mode: str):
    """
    -> logits (M, C); labels (M,) [multiclass] or (M, C) [multilabel].
    Accepts (N, C), (N, C, *spatial), and matching label shapes.
    """
    logits = logits.detach().float()
    labels = labels.detach()
    if logits.ndim > 2:                       # (N, C, *spatial) -> (M, C)
        c = logits.shape[1]
        logits = logits.movedim(1, -1).reshape(-1, c)
        if mode == "multilabel":
            labels = labels.movedim(1, -1).reshape(-1, c)
        else:
            labels = labels.reshape(-1)
    return logits, labels


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------
def fit_temperature(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    mode: str = "multiclass",
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """
    Fit a single temperature T by minimising validation loss (NLL for
    multiclass, BCE for multilabel). Optimises log(T) so T stays positive.

    Returns the learned T (pass it to :func:`apply_temperature` or into
    ``ConfidenceLocalizer(temperature=T)``).
    """
    if mode not in ("multiclass", "multilabel"):
        raise ValueError("mode must be 'multiclass' or 'multilabel'")
    logits, labels = _flatten(val_logits, val_labels, mode)
    labels = labels.long() if mode == "multiclass" else labels.float()

    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        scaled = logits / log_T.exp()
        loss = (F.cross_entropy(scaled, labels) if mode == "multiclass"
                else F.binary_cross_entropy_with_logits(scaled, labels))
        loss.backward()
        return loss

    opt.step(_closure)
    T = float(log_T.exp().item())
    logger.info("Fitted temperature T=%.4f (%s)", T, mode)
    return T


def apply_temperature(logits: torch.Tensor, temperature: float,
                      mode: str = "multiclass") -> torch.Tensor:
    """Scale logits by T and convert to probabilities."""
    scaled = logits / float(temperature)
    if mode == "multiclass":
        return F.softmax(scaled, dim=-1 if scaled.ndim == 1 else 1)
    return torch.sigmoid(scaled)


# --------------------------------------------------------------------------
# Expected Calibration Error — the proof metric
# --------------------------------------------------------------------------
def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "multiclass",
    n_bins: int = 15,
) -> float:
    """
    ECE: average gap between confidence and accuracy across probability bins.
    Lower is better (0 = perfectly calibrated).

    * multiclass: confidence = max prob; accuracy = argmax == label.
    * multilabel: flatten (sample, class); confidence = predicted prob of
      positive; accuracy = empirical positive frequency in the bin.
    """
    probs = probs.detach().float()
    labels = labels.detach()

    if mode == "multiclass":
        if probs.ndim > 2:
            probs = probs.movedim(1, -1).reshape(-1, probs.shape[1])
            labels = labels.reshape(-1)
        conf, pred = probs.max(dim=1)
        correct = (pred == labels.long()).float()
    else:  # multilabel / binary — flatten everything
        conf = probs.reshape(-1)
        correct = labels.reshape(-1).float()   # bin "accuracy" = P(positive)

    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = conf.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            bin_conf = conf[mask].mean().item()
            bin_acc = correct[mask].mean().item()
            ece += (mask.float().mean().item()) * abs(bin_conf - bin_acc)
    return float(ece)


# --------------------------------------------------------------------------
# persistence — per-model temperatures actually used at inference
# --------------------------------------------------------------------------
class CalibrationStore:
    """
    Maps model key -> fitted temperature, persisted as JSON. Unknown keys
    return 1.0 (a no-op), so inference is always safe even before calibration.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self._t: Dict[str, float] = {}
        if self.path and self.path.exists():
            try:
                self._t = {k: float(v) for k, v in
                           json.loads(self.path.read_text()).items()}
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not read calibration store %s: %s",
                               self.path, exc)

    def get(self, key: str) -> float:
        return self._t.get(key, 1.0)

    def set(self, key: str, temperature: float) -> None:
        self._t[key] = float(temperature)
        if self.path:
            self.path.write_text(json.dumps(self._t, indent=2))

    def as_dict(self) -> Dict[str, float]:
        return dict(self._t)


def calibrate_and_store(
    model_key: str,
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    mode: str,
    store: CalibrationStore,
    n_bins: int = 15,
) -> dict:
    """
    Offline workflow: fit T on a model's validation logits, record it in the
    store (so inference applies it), and report the before/after ECE so the
    improvement is auditable. Returns a summary dict.
    """
    probs_before = apply_temperature(val_logits, 1.0, mode)
    ece_before = expected_calibration_error(probs_before, val_labels, mode, n_bins)
    T = fit_temperature(val_logits, val_labels, mode)
    ece_after = expected_calibration_error(
        apply_temperature(val_logits, T, mode), val_labels, mode, n_bins)
    store.set(model_key, T)
    summary = {"model_key": model_key, "temperature": round(T, 4),
               "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4)}
    logger.info("Calibrated %s: %s", model_key, summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    # Over-confident multiclass logits: true class present but logits too sharp.
    N, Cc = 4000, 5
    labels = torch.randint(0, Cc, (N,))
    base = torch.randn(N, Cc)
    base[torch.arange(N), labels] += 1.2          # modest signal
    logits = base * 3.0                            # inflate -> over-confident
    before = expected_calibration_error(F.softmax(logits, 1), labels, "multiclass")
    T = fit_temperature(logits, labels, "multiclass")
    after = expected_calibration_error(apply_temperature(logits, T, "multiclass"),
                                       labels, "multiclass")
    print(f"multiclass  T={T:.3f}  ECE {before:.4f} -> {after:.4f}")
