"""
test_coordinate_mapping.py
==================================================================
Validates coordinate mapping with analytically-known answers and a real
ingested-scan round-trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from coordinate_mapping import (                              # noqa: E402
    divisible_pad_before, rescale_box, uncrop_box,
    voxel_box_to_world, voxel_box_between_grids, map_detection_bbox,
    bbox_flat_to_lohi,
)


def approx(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


def test_pad_matches_monai() -> bool:
    from monai.transforms import DivisiblePad
    ok = True
    for s in [(45, 45, 8), (37, 37, 37), (50, 64, 33)]:
        x = torch.zeros(1, *s)
        padded = DivisiblePad(k=16)(x)
        # empirical pad-before per axis via first nonzero of an all-ones pad
        ones = DivisiblePad(k=16)(torch.ones(1, *s))[0]  # drop channel -> N-D
        emp = []
        for ax in range(len(s)):
            axes = tuple(i for i in range(len(s)) if i != ax)
            nz = (ones.sum(dim=axes) > 0).nonzero().flatten()
            emp.append(int(nz[0]))
        pred = divisible_pad_before(s, 16)
        ok &= approx(pred, emp)
    print(f"  pad_before matches MONAI: {'PASS' if ok else 'FAIL'}")
    return ok


def test_rescale() -> bool:
    # feature grid 16 -> input grid 256 (x16). box [4,7] -> [64, 127].
    lo, hi = rescale_box(np.array([4.]), np.array([7.]), [16], [256])
    ok = approx(lo, [64.]) and approx(hi, [127.])
    print(f"  rescale feature->input: {'PASS' if ok else 'FAIL'} "
          f"([4,7]@16 -> [{lo[0]:.0f},{hi[0]:.0f}]@256)")
    return ok


def test_affine_roundtrip() -> bool:
    # Realistic flip+scale affines (RAS-style): diagonal with a sign flip.
    A0 = np.diag([-0.7, 0.7, 1.0, 1.0]); A0[:3, 3] = [10, -5, 3]
    A1 = np.diag([-1.0, 1.0, 1.0, 1.0]); A1[:3, 3] = [12, -6, 3]
    lo, hi = np.array([5., 8., 2.]), np.array([20., 25., 6.])
    w_direct = voxel_box_to_world(lo, hi, A1)
    olo, ohi = voxel_box_between_grids(lo, hi, A1, A0)
    w_via = voxel_box_to_world(olo, ohi, A0)
    ok = approx(w_direct[0], w_via[0]) and approx(w_direct[1], w_via[1])
    print(f"  affine two-stage == direct (flip+scale): {'PASS' if ok else 'FAIL'}")
    return ok


def test_axis_permutation() -> bool:
    # A1 swaps x/y vs A0 -> mapping must swap the box axes accordingly.
    A0 = np.diag([1., 1., 1., 1.])
    A1 = np.array([[0., 1., 0., 0.],
                   [1., 0., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])  # swap axis 0 and 1
    lo, hi = np.array([2., 10., 0.]), np.array([4., 20., 5.])
    olo, ohi = voxel_box_between_grids(lo, hi, A1, A0)
    ok = approx(sorted([olo[0], ohi[0]]), [10., 20.]) and \
         approx(sorted([olo[1], ohi[1]]), [2., 4.])
    print(f"  axis permutation handled: {'PASS' if ok else 'FAIL'}")
    return ok


def test_real_scan_roundtrip() -> bool:
    from synthetic_dicom import make_series
    from dicom_ingestion import load_and_preprocess
    import tempfile
    d = make_series(tempfile.mkdtemp() + "/ct", "CT", "CHEST", volumetric=True)
    scan = load_and_preprocess(d)
    S1 = scan.preprocessed_shape           # e.g. (45, 45, 8)
    S0 = scan.original_shape               # e.g. (64, 64, 8)

    # Full preprocessed-volume box should map to ~the full original extent.
    full = tuple(v for s in S1 for v in (0, s - 1))
    res = map_detection_bbox(
        full,
        preprocessed_affine=scan.preprocessed_affine,
        original_affine=scan.original_affine,
        clip_shape=S0,
    )
    lo, hi = bbox_flat_to_lohi(res["original_voxel"])
    span = hi - lo + 1
    covers = all(span[i] >= 0.8 * S0[i] for i in range(len(S0)))
    inside = all(0 <= lo[i] and hi[i] <= S0[i] - 1 for i in range(len(S0)))
    has_world = "world_mm" in res
    ok = covers and inside and has_world
    print(f"  real CT: S1={S1} S0={S0}")
    print(f"           full box -> original_voxel={res['original_voxel']} "
          f"(covers>=80%: {covers}, in-bounds: {inside})")
    print(f"           world_mm={res.get('world_mm')}")
    print(f"  real-scan round-trip: {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradcam_style_compose() -> bool:
    # Grad-CAM box on a 5x5x1 feature grid of a 45x45x8 input, identity affine.
    A = np.eye(4)
    res = map_detection_bbox(
        (1, 3, 1, 3, 0, 0),
        feature_shape=(5, 5, 1), input_shape=(45, 45, 8),
        preprocessed_affine=A, original_affine=A, clip_shape=(45, 45, 8),
    )
    lo, hi = bbox_flat_to_lohi(res["input_voxel"])
    # axis0: [1,3]@5 -> [9, 35]@45 (1*9=9 ; (3+1)*9-1=35)
    ok = approx(lo[:2], [9., 9.]) and approx(hi[:2], [35., 35.])
    print(f"  grad-cam feature->input compose: {'PASS' if ok else 'FAIL'} "
          f"(input_voxel={res['input_voxel']})")
    return ok


def main() -> int:
    print("=" * 62)
    print("COORDINATE MAPPING TESTS")
    print("=" * 62)
    results = [
        test_pad_matches_monai(),
        test_rescale(),
        test_affine_roundtrip(),
        test_axis_permutation(),
        test_gradcam_style_compose(),
        test_real_scan_roundtrip(),
    ]
    print("=" * 62)
    ok = all(results)
    print(f"RESULT: {'ALL GREEN' if ok else 'FAILURE'} "
          f"({sum(results)}/{len(results)})")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
