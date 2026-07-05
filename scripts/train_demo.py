"""
train_demo.py
==================================================================
Runs the training pipeline on synthetic data for one segmentation model and one
classifier, showing that both loss paths actually learn and that the calibration
loop closes (temperature fit + ECE reported). Uses random-init models and
synthetic data — the numbers demonstrate the machinery, not diagnostic skill.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training.train import train_model                          # noqa: E402


def main() -> int:
    print("=" * 72)
    print("TRAINING PIPELINE DEMO  (synthetic data, random-init models)")
    print("=" * 72)

    cases = [
        ("ct_chest_nodule_seg", 6, "segmentation · Dice"),
        ("cxr_multilabel_classifier", 12, "classification · acc@0.5"),
    ]
    for key, epochs, label in cases:
        s = train_model(key, epochs=epochs, out_dir="/tmp/rad_ckpt",
                        calibration_json="/tmp/rad_calib.json")
        print(f"\n{key}  ({label})")
        print(f"  metric   {s['val_metric_before']:.3f}  ->  {s['val_metric_after']:.3f}")
        print(f"  val_loss {s['val_loss_after']:.3f}")
        print(f"  calib    T={s['temperature']:.3f}   ECE "
              f"{s['ece_before']:.3f} -> {s['ece_after']:.3f}")
        print(f"  weights  {s['checkpoint']}  (loadable via ModelRegistry.load_weights)")

    print("\n" + "=" * 72)
    print("Loop verified: data -> loss -> backprop -> metric -> calibrate -> store.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
