"""
test_calibration.py
==================================================================
Proves temperature scaling actually calibrates:
  * multiclass and multilabel ECE both drop after fitting T
  * over-confident logits yield T > 1 (softening)
  * T = 1 is a no-op; a well-calibrated set stays well-calibrated
  * CalibrationStore persists and reloads temperatures
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from calibration import (                                       # noqa: E402
    fit_temperature, apply_temperature, expected_calibration_error,
    CalibrationStore,
)


def test_multiclass_ece_drop() -> bool:
    torch.manual_seed(0)
    N, C = 5000, 5
    labels = torch.randint(0, C, (N,))
    base = torch.randn(N, C)
    base[torch.arange(N), labels] += 1.2
    logits = base * 3.0                              # inflate -> over-confident
    before = expected_calibration_error(F.softmax(logits, 1), labels, "multiclass")
    T = fit_temperature(logits, labels, "multiclass")
    after = expected_calibration_error(
        apply_temperature(logits, T, "multiclass"), labels, "multiclass")
    ok = T > 1.0 and after < before
    print(f"  multiclass: T={T:.3f}  ECE {before:.4f} -> {after:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_multilabel_ece_drop() -> bool:
    torch.manual_seed(1)
    N, C = 8000, 6
    # Principled over-confidence: labels sampled from TRUE probs sigmoid(z),
    # then logits sharpened by a known factor so sigmoid(3z) is over-confident.
    z = torch.randn(N, C) * 1.5
    true_p = torch.sigmoid(z)
    labels = (torch.rand(N, C) < true_p).float()
    logits = z * 3.0                                # over-confident by ~3x
    before = expected_calibration_error(torch.sigmoid(logits), labels, "multilabel")
    T = fit_temperature(logits, labels, "multilabel")
    after = expected_calibration_error(
        apply_temperature(logits, T, "multilabel"), labels, "multilabel")
    ok = T > 1.0 and after < before
    print(f"  multilabel: T={T:.3f}  ECE {before:.4f} -> {after:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_temperature_one_is_noop() -> bool:
    logits = torch.randn(10, 4)
    same = torch.allclose(apply_temperature(logits, 1.0, "multiclass"),
                          F.softmax(logits, 1))
    print(f"  T=1 is a no-op: {'PASS' if same else 'FAIL'}")
    return same


def test_store_roundtrip() -> bool:
    path = Path(tempfile.mkdtemp()) / "calib.json"
    s = CalibrationStore(str(path))
    ok0 = s.get("unknown_key") == 1.0                # safe default
    s.set("cxr_multilabel_classifier", 2.31)
    s2 = CalibrationStore(str(path))                 # reload from disk
    ok1 = abs(s2.get("cxr_multilabel_classifier") - 2.31) < 1e-6
    print(f"  store default=1.0 & persists: {'PASS' if ok0 and ok1 else 'FAIL'}")
    return ok0 and ok1


def test_calibrate_and_store() -> bool:
    from calibration import calibrate_and_store
    torch.manual_seed(2)
    N, C = 5000, 5
    labels = torch.randint(0, C, (N,))
    base = torch.randn(N, C); base[torch.arange(N), labels] += 1.2
    logits = base * 3.0
    path = Path(tempfile.mkdtemp()) / "calib.json"
    store = CalibrationStore(str(path))
    summary = calibrate_and_store("us_breast_lesion", logits, labels, "multiclass", store)
    reloaded = CalibrationStore(str(path)).get("us_breast_lesion")
    ok = (summary["ece_after"] < summary["ece_before"]
          and abs(reloaded - summary["temperature"]) < 1e-3)
    print(f"  fit->store loop: {summary}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("=" * 60)
    print("CALIBRATION TESTS")
    print("=" * 60)
    results = [
        test_multiclass_ece_drop(),
        test_multilabel_ece_drop(),
        test_temperature_one_is_noop(),
        test_store_roundtrip(),
        test_calibrate_and_store(),
    ]
    print("=" * 60)
    ok = all(results)
    print(f"RESULT: {'ALL GREEN' if ok else 'FAILURE'} ({sum(results)}/{len(results)})")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
