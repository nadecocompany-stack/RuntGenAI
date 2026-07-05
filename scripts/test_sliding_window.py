"""
test_sliding_window.py
==================================================================
Validates memory-bounded volumetric inference:
  * default ROI selection (rounds to stride multiple, capped at base)
  * a small volume returns full-resolution output (ROI pads internally)
  * a LARGE volume that would be costly in one pass tiles into windows and
    still returns full-resolution output (peak memory stays at patch size)
  * autocast (AMP) path runs
  * determinism (same input -> same output)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import segment_volume, default_roi_size            # noqa: E402
from monai.networks.nets import UNet                              # noqa: E402


def _net():
    return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2).eval()


def test_roi_selection() -> bool:
    cases = {(45, 45, 8): (48, 48, 16), (512, 512, 300): (96, 96, 96),
             (160, 160, 64): (96, 96, 64)}
    ok = all(default_roi_size(k) == v for k, v in cases.items())
    print(f"  ROI selection: {'PASS' if ok else 'FAIL'} "
          f"(e.g. (512,512,300)->{default_roi_size((512,512,300))})")
    return ok


def test_small_fullres() -> bool:
    net = _net()
    x = torch.randn(1, 1, 45, 45, 8)
    y = segment_volume(net, x, device="cpu")
    ok = tuple(y.shape) == (1, 2, 45, 45, 8)
    print(f"  small volume full-res: {'PASS' if ok else 'FAIL'} -> {tuple(y.shape)}")
    return ok


def test_large_tiled() -> bool:
    net = _net()
    # A volume large enough to force many windows; whole-volume forward here
    # would allocate a big activation map — sliding window keeps it patch-sized.
    x = torch.randn(1, 1, 256, 256, 40)
    t = time.time()
    y = segment_volume(net, x, roi_size=(96, 96, 32), overlap=0.25, device="cpu")
    dt = time.time() - t
    ok = tuple(y.shape[2:]) == (256, 256, 40)
    print(f"  large volume tiled: {'PASS' if ok else 'FAIL'} "
          f"-> {tuple(y.shape[2:])} in {dt:.1f}s")
    return ok


def test_amp_runs() -> bool:
    net = _net()
    x = torch.randn(1, 1, 64, 64, 16)
    try:
        y = segment_volume(net, x, use_amp=True, device="cpu")
        ok = tuple(y.shape[2:]) == (64, 64, 16)
    except Exception as exc:  # noqa: BLE001
        print(f"    AMP error: {exc}")
        ok = False
    print(f"  autocast path runs: {'PASS' if ok else 'FAIL'}")
    return ok


def test_deterministic() -> bool:
    net = _net()
    x = torch.randn(1, 1, 48, 48, 16)
    a = segment_volume(net, x, device="cpu")
    b = segment_volume(net, x, device="cpu")
    ok = torch.allclose(a, b, atol=1e-5)
    print(f"  deterministic: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("=" * 60)
    print("SLIDING-WINDOW INFERENCE TESTS")
    print("=" * 60)
    results = [
        test_roi_selection(),
        test_small_fullres(),
        test_large_tiled(),
        test_amp_runs(),
        test_deterministic(),
    ]
    print("=" * 60)
    ok = all(results)
    print(f"RESULT: {'ALL GREEN' if ok else 'FAILURE'} ({sum(results)}/{len(results)})")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
