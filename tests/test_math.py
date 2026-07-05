"""Coordinate mapping, calibration, and sliding-window inference tests."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# coordinate mapping
# --------------------------------------------------------------------------
def test_rescale_feature_to_input():
    from coordinate_mapping import rescale_box
    lo, hi = rescale_box(np.array([4.]), np.array([7.]), [16], [256])
    assert np.allclose(lo, [64.]) and np.allclose(hi, [127.])


def test_affine_two_stage_equals_direct():
    from coordinate_mapping import voxel_box_to_world, voxel_box_between_grids
    A0 = np.diag([-0.7, 0.7, 1.0, 1.0]); A0[:3, 3] = [10, -5, 3]
    A1 = np.diag([-1.0, 1.0, 1.0, 1.0]); A1[:3, 3] = [12, -6, 3]
    lo, hi = np.array([5., 8., 2.]), np.array([20., 25., 6.])
    wd = voxel_box_to_world(lo, hi, A1)
    olo, ohi = voxel_box_between_grids(lo, hi, A1, A0)
    wv = voxel_box_to_world(olo, ohi, A0)
    assert np.allclose(wd[0], wv[0]) and np.allclose(wd[1], wv[1])


def test_axis_permutation():
    from coordinate_mapping import voxel_box_between_grids
    A0 = np.eye(4)
    A1 = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.],
                   [0., 0., 1., 0.], [0., 0., 0., 1.]])
    lo, hi = np.array([2., 10., 0.]), np.array([4., 20., 5.])
    olo, ohi = voxel_box_between_grids(lo, hi, A1, A0)
    assert sorted([olo[0], ohi[0]]) == [10., 20.]
    assert sorted([olo[1], ohi[1]]) == [2., 4.]


def test_pad_before_matches_monai():
    from coordinate_mapping import divisible_pad_before
    from monai.transforms import DivisiblePad
    for s in [(45, 45, 8), (37, 37, 37)]:
        ones = DivisiblePad(k=16)(torch.ones(1, *s))[0]
        emp = [int((ones.sum(dim=tuple(j for j in range(len(s)) if j != ax)) > 0)
                   .nonzero().flatten()[0]) for ax in range(len(s))]
        assert list(divisible_pad_before(s, 16)) == emp


def test_real_scan_roundtrip(ct_series):
    from dicom_ingestion import load_and_preprocess
    from coordinate_mapping import map_detection_bbox, bbox_flat_to_lohi
    scan = load_and_preprocess(ct_series)
    S1, S0 = scan.preprocessed_shape, scan.original_shape
    full = tuple(v for s in S1 for v in (0, s - 1))
    res = map_detection_bbox(full, preprocessed_affine=scan.preprocessed_affine,
                             original_affine=scan.original_affine, clip_shape=S0)
    lo, hi = bbox_flat_to_lohi(res["original_voxel"])
    span = hi - lo + 1
    assert all(span[i] >= 0.8 * S0[i] for i in range(len(S0)))
    assert "world_mm" in res


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def test_calibration_multiclass_reduces_ece():
    from calibration import fit_temperature, apply_temperature, expected_calibration_error
    N, C = 4000, 5
    labels = torch.randint(0, C, (N,))
    base = torch.randn(N, C); base[torch.arange(N), labels] += 1.2
    logits = base * 3.0
    before = expected_calibration_error(F.softmax(logits, 1), labels, "multiclass")
    T = fit_temperature(logits, labels, "multiclass")
    after = expected_calibration_error(apply_temperature(logits, T, "multiclass"),
                                       labels, "multiclass")
    assert T > 1.0 and after < before


def test_calibration_multilabel_reduces_ece():
    from calibration import fit_temperature, apply_temperature, expected_calibration_error
    N, C = 6000, 6
    z = torch.randn(N, C) * 1.5
    labels = (torch.rand(N, C) < torch.sigmoid(z)).float()
    logits = z * 3.0
    before = expected_calibration_error(torch.sigmoid(logits), labels, "multilabel")
    T = fit_temperature(logits, labels, "multilabel")
    after = expected_calibration_error(apply_temperature(logits, T, "multilabel"),
                                       labels, "multilabel")
    assert T > 1.0 and after < before


def test_calibration_store_roundtrip(tmp_path):
    from calibration import CalibrationStore
    p = tmp_path / "calib.json"
    s = CalibrationStore(str(p))
    assert s.get("missing") == 1.0
    s.set("k", 2.3)
    assert abs(CalibrationStore(str(p)).get("k") - 2.3) < 1e-6


# --------------------------------------------------------------------------
# sliding-window inference
# --------------------------------------------------------------------------
def test_roi_selection():
    from inference import default_roi_size
    assert default_roi_size((45, 45, 8)) == (48, 48, 16)
    assert default_roi_size((512, 512, 300)) == (96, 96, 96)


def test_segment_full_resolution(small_unet):
    from inference import segment_volume
    x = torch.randn(1, 1, 45, 45, 8)
    y = segment_volume(small_unet, x, device="cpu")
    assert tuple(y.shape) == (1, 2, 45, 45, 8)


def test_segment_large_tiled(small_unet):
    from inference import segment_volume
    x = torch.randn(1, 1, 128, 128, 32)
    y = segment_volume(small_unet, x, roi_size=(96, 96, 32), device="cpu")
    assert tuple(y.shape[2:]) == (128, 128, 32)


def test_segment_deterministic(small_unet):
    from inference import segment_volume
    x = torch.randn(1, 1, 48, 48, 16)
    a = segment_volume(small_unet, x, device="cpu")
    b = segment_volume(small_unet, x, device="cpu")
    assert torch.allclose(a, b, atol=1e-5)
