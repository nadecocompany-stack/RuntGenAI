"""
coordinate_mapping.py
==================================================================
Map bounding boxes from the space a localizer ran in back to clinically useful
coordinates: original scan voxels and patient/world millimetres.

Why several stages
------------------
A detection is produced in one of these spaces:
  * Grad-CAM feature grid  (downsampled: /stride for CNNs, /patch for ViTs)
  * padded preprocessed volume  (after DivisiblePad for U-Nets)
  * preprocessed volume  (after Orientation + Spacing resampling)
None of these is the space the scan was uploaded in. Mapping back composes:

  feature grid --rescale--> (padded) preprocessed --uncrop--> preprocessed
              --affine(inv(A0)·A1)--> original voxels --A0--> world (mm)

Orientation flips and anisotropic resampling are both encoded in the MetaTensor
affines, so ``inv(A_original) @ A_preprocessed`` handles them together — no need
to track axis permutations by hand.

Boxes are represented as ``(lo, hi)`` inclusive corner arrays (length N). Helpers
convert to/from the flat ``Detection.bbox`` tuple
``(a0_min, a0_max, a1_min, a1_max, ...)``.
"""

from __future__ import annotations

import itertools
from typing import Sequence, Tuple

import numpy as np

Vec = np.ndarray


# --------------------------------------------------------------------------
# flat <-> (lo, hi)
# --------------------------------------------------------------------------
def bbox_flat_to_lohi(bbox: Sequence[int]) -> Tuple[Vec, Vec]:
    """(a0_min,a0_max,a1_min,a1_max,...) -> (lo, hi)."""
    arr = np.asarray(bbox, dtype=float)
    if arr.size % 2 != 0:
        raise ValueError(f"bbox length must be even, got {arr.size}")
    lo = arr[0::2]
    hi = arr[1::2]
    return lo, hi


def lohi_to_bbox_flat(lo: Vec, hi: Vec) -> Tuple[int, ...]:
    out: list = []
    for a, b in zip(lo, hi):
        out.extend((int(round(a)), int(round(b))))
    return tuple(out)


# --------------------------------------------------------------------------
# stage helpers
# --------------------------------------------------------------------------
def divisible_pad_before(shape: Sequence[int], k: int) -> Vec:
    """
    Per-axis pad-before that ``monai.transforms.DivisiblePad(k, "symmetric")``
    adds: total = (k - s % k) % k, before = total // 2 (verified against MONAI).
    """
    arr = np.asarray(shape, dtype=int)
    total = (k - arr % k) % k
    return total // 2


def rescale_box(lo: Vec, hi: Vec, src_shape: Sequence[int],
                dst_shape: Sequence[int]) -> Tuple[Vec, Vec]:
    """
    Linearly rescale a box between two grid resolutions (e.g. Grad-CAM feature
    grid -> input grid). Scales corner coordinates by dst/src per axis.
    """
    src = np.asarray(src_shape, dtype=float)
    dst = np.asarray(dst_shape, dtype=float)
    scale = dst / src
    return lo * scale, (hi + 1.0) * scale - 1.0   # keep inclusive-max semantics


def uncrop_box(lo: Vec, hi: Vec, pad_before: Sequence[int]) -> Tuple[Vec, Vec]:
    """Undo a symmetric pad: subtract the per-axis pad-before offset."""
    off = np.asarray(pad_before, dtype=float)
    return lo - off, hi - off


# --------------------------------------------------------------------------
# affine corner mapping
# --------------------------------------------------------------------------
def reduce_affine_to_ndim(affine: Vec, n: int) -> Vec:
    """
    Reduce an (M+1)x(M+1) affine to (n+1)x(n+1), keeping the first ``n`` spatial
    axes and the homogeneous translation column. Needed because MONAI stores a
    4x4 affine even for 2D images; a 2D box must be mapped with a 3x3 affine
    (otherwise the translation is read from the wrong column).
    """
    affine = np.asarray(affine, dtype=float)
    m = affine.shape[0] - 1
    if m == n:
        return affine
    idx = list(range(n)) + [m]          # spatial axes 0..n-1 + homogeneous row/col
    return affine[np.ix_(idx, idx)]


def _corners(lo: Vec, hi: Vec) -> Vec:
    """All 2^N corners of the box, shape (2^N, N)."""
    return np.array(list(itertools.product(*zip(lo, hi))), dtype=float)


def _apply_affine(points: Vec, affine: Vec) -> Vec:
    """Apply an (N+1)x(N+1) affine to (M, N) points -> (M, N)."""
    n = points.shape[1]
    R = affine[:n, :n]
    t = affine[:n, n]
    return points @ R.T + t


def voxel_box_to_world(lo: Vec, hi: Vec, affine: Vec) -> Tuple[Vec, Vec]:
    """
    Map a voxel-index box to an axis-aligned world (mm) box via ``affine``
    (voxel -> world). Uses all corners so axis flips/rotations are respected;
    the result is the tight axis-aligned bound in world space.
    """
    w = _apply_affine(_corners(lo, hi), affine)
    return w.min(axis=0), w.max(axis=0)


def voxel_box_between_grids(lo: Vec, hi: Vec, affine_src: Vec,
                            affine_dst: Vec) -> Tuple[Vec, Vec]:
    """
    Move a voxel box from a source grid to a destination grid that share the
    same world space: compose ``inv(affine_dst) @ affine_src`` and map corners.
    Handles resampling scale and orientation flips in one step.
    """
    m = np.linalg.inv(affine_dst) @ affine_src
    p = _apply_affine(_corners(lo, hi), m)
    return p.min(axis=0), p.max(axis=0)


# --------------------------------------------------------------------------
# high-level composition
# --------------------------------------------------------------------------
def map_detection_bbox(
    bbox: Sequence[int],
    *,
    feature_shape: Sequence[int] | None = None,
    input_shape: Sequence[int] | None = None,
    pad_before: Sequence[int] | None = None,
    preprocessed_affine: Vec | None = None,
    original_affine: Vec | None = None,
    clip_shape: Sequence[int] | None = None,
) -> dict:
    """
    Compose the full mapping and return coordinates in every available space.

    Parameters (all optional — apply only the stages you need)
    ----------
    feature_shape / input_shape : if the box is on a Grad-CAM feature grid,
        rescale from feature_shape to input_shape first.
    pad_before : per-axis DivisiblePad offset to remove (U-Net path).
    preprocessed_affine (A1) / original_affine (A0) : MetaTensor affines. If both
        given, also return original-voxel and world-mm boxes.
    clip_shape : clamp original-voxel box into [0, shape) per axis.

    Returns dict with keys among: 'input_voxel', 'original_voxel', 'world_mm'.
    """
    lo, hi = bbox_flat_to_lohi(bbox)
    n = len(lo)

    if feature_shape is not None and input_shape is not None:
        lo, hi = rescale_box(lo, hi, feature_shape, input_shape)
    if pad_before is not None:
        lo, hi = uncrop_box(lo, hi, pad_before)

    out: dict = {"input_voxel": lohi_to_bbox_flat(lo, hi)}

    A1 = reduce_affine_to_ndim(preprocessed_affine, n) if preprocessed_affine is not None else None
    A0 = reduce_affine_to_ndim(original_affine, n) if original_affine is not None else None

    if A1 is not None and A0 is not None:
        olo, ohi = voxel_box_between_grids(lo, hi, A1, A0)
        if clip_shape is not None:
            cs = np.asarray(clip_shape, float) - 1.0
            olo = np.clip(np.minimum(olo, ohi), 0, cs)
            ohi = np.clip(np.maximum(olo, ohi), 0, cs)
        out["original_voxel"] = lohi_to_bbox_flat(olo, ohi)
        wlo, whi = voxel_box_to_world(olo, ohi, A0)
        out["world_mm"] = tuple(
            round(float(v), 2) for pair in zip(wlo, whi) for v in pair
        )
    elif A1 is not None:
        wlo, whi = voxel_box_to_world(lo, hi, A1)
        out["world_mm"] = tuple(
            round(float(v), 2) for pair in zip(wlo, whi) for v in pair
        )
    return out
