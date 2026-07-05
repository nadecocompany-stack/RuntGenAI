"""
dicom_ingestion.py
==================================================================
Robust DICOM -> normalized PyTorch tensor ingestion pipeline.

Responsibilities
----------------
1. Scan a directory of DICOM files, tolerating corrupted / non-image files.
2. Extract routing metadata (Modality, BodyPartExamined) via pydicom.
3. Assemble slices into a correctly ordered 3D volume (or single 2D frame).
4. Standardise geometry (voxel spacing / orientation) with MONAI transforms.
5. Apply modality-appropriate intensity normalisation:
      - CT  -> Hounsfield-Unit windowing (rescale slope/intercept + W/L clip)
      - MR  -> robust percentile / z-score normalisation (HU is meaningless)
      - US / XR / other -> min-max to [0, 1]
6. Return an inference-ready tensor plus a metadata record used for routing.

Design notes
------------
* pydicom owns *metadata* and per-file corruption handling; MONAI's
  ``PydicomReader`` owns *pixel + affine* assembly so voxel geometry stays
  consistent with the DICOM Image Plane Module. This is the split the RAD-DINO
  authors also use (SimpleITK/pydicom) and it avoids hand-rolling affines.
* Nothing here makes a clinical decision — this is pure geometry + tensor
  normalisation. Downstream models consume ``PreprocessedScan.tensor``.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

# MONAI array (non-dictionary) transforms operate directly on numpy/MetaTensor.
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Orientation,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    Spacing,
    ToTensor,
)
from monai.data import MetaTensor, PydicomReader

logger = logging.getLogger("dicom_ingestion")
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Routing enums + config
# ---------------------------------------------------------------------------
class Modality(str, Enum):
    """Normalised modality tokens. DICOM tag (0008,0060) uses these codes."""
    CT = "CT"
    MR = "MR"
    US = "US"
    XR = "XR"          # covers CR / DX / RF projection radiography
    UNKNOWN = "UNKNOWN"


# DICOM's Modality value uses CR/DX/RF for projection X-ray; fold them together.
_MODALITY_ALIASES: Dict[str, Modality] = {
    "CT": Modality.CT,
    "MR": Modality.MR, "MRI": Modality.MR,
    "US": Modality.US,
    "CR": Modality.XR, "DX": Modality.XR, "RF": Modality.XR, "XR": Modality.XR,
}

# CT window presets as (window_width, window_level) in HU, keyed by body part.
# These control *display/normalisation contrast*, not diagnosis. Extend freely.
CT_WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (400.0, 40.0),      # soft tissue / abdomen
    "CHEST":   (1500.0, -600.0),   # lung window
    "LUNG":    (1500.0, -600.0),
    "ABDOMEN": (400.0, 40.0),
    "BRAIN":   (80.0, 40.0),
    "BONE":    (1800.0, 400.0),
    "HEAD":    (80.0, 40.0),
}


@dataclass
class ScanMetadata:
    """Routing + provenance metadata extracted from the DICOM header."""
    modality: Modality
    body_part: str
    study_uid: Optional[str]
    series_uid: Optional[str]
    n_slices: int
    original_spacing: Optional[Tuple[float, float, float]]
    rescale_slope: float
    rescale_intercept: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedScan:
    """Container returned to the inference router."""
    tensor: "MetaTensor"            # shape (C, [D,] H, W), float32, normalised
    metadata: ScanMetadata
    file_paths: List[str]
    # Geometry for mapping detections back to original scan / world space.
    original_affine: Optional[np.ndarray] = None      # A0: original voxel -> world
    preprocessed_affine: Optional[np.ndarray] = None  # A1: preprocessed voxel -> world
    original_shape: Optional[Tuple[int, ...]] = None   # spatial dims, pre-transform
    preprocessed_shape: Optional[Tuple[int, ...]] = None  # spatial dims, post-transform


# ---------------------------------------------------------------------------
# Step 1 — metadata scan (pydicom, corruption-tolerant)
# ---------------------------------------------------------------------------
def _safe_read_header(path: Path) -> Optional[pydicom.dataset.FileDataset]:
    """
    Read a DICOM header without pixel data. Returns None for anything that
    is not a usable DICOM file so the caller can skip it silently.

    We defensively catch:
      * InvalidDicomError  -> missing/valid preamble but bad SOP structure
      * OSError / EOFError -> truncated files, permission issues
      * Exception          -> pydicom can still raise on exotic private tags
    ``force=True`` lets us recover files that lack the 128-byte preamble but
    are otherwise valid (a common export quirk).
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except (InvalidDicomError, OSError, EOFError) as exc:
        logger.debug("Skipping non-DICOM/unreadable file %s (%s)", path.name, exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.warning("Unexpected error reading header %s: %s", path.name, exc)
        return None

    # A real image object must at least declare a Modality. Files that don't
    # (DICOMDIR, structured reports, presentation states) are not volumes.
    if "Modality" not in ds:
        return None
    return ds


def _normalise_modality(raw: Optional[str]) -> Modality:
    if not raw:
        return Modality.UNKNOWN
    return _MODALITY_ALIASES.get(str(raw).strip().upper(), Modality.UNKNOWN)


def scan_series(dicom_dir: str | Path) -> Tuple[List[Path], ScanMetadata]:
    """
    Walk ``dicom_dir``, collect valid image slices belonging to the *largest*
    series present (guards against mixed studies in one folder), and build a
    :class:`ScanMetadata` record from the first valid slice.

    Raises
    ------
    FileNotFoundError : directory missing.
    ValueError        : no valid DICOM image slices found.
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Group valid slices by SeriesInstanceUID; keep the largest coherent series.
    series_buckets: Dict[str, List[Tuple[Path, pydicom.dataset.FileDataset]]] = {}
    for path in sorted(dicom_dir.rglob("*")):
        if not path.is_file():
            continue
        ds = _safe_read_header(path)
        if ds is None:
            continue
        series_uid = getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")
        series_buckets.setdefault(series_uid, []).append((path, ds))

    if not series_buckets:
        raise ValueError(f"No valid DICOM image slices found in {dicom_dir}")

    series_uid, slices = max(series_buckets.items(), key=lambda kv: len(kv[1]))
    if len(series_buckets) > 1:
        logger.warning(
            "Multiple series in %s; using largest (%d slices, uid=%s)",
            dicom_dir, len(slices), series_uid,
        )

    ref_ds = slices[0][1]
    modality = _normalise_modality(getattr(ref_ds, "Modality", None))

    # PixelSpacing = [row_spacing, col_spacing] (mm); SliceThickness = z (mm).
    px = getattr(ref_ds, "PixelSpacing", None)
    thickness = getattr(ref_ds, "SliceThickness", None)
    if px is not None:
        try:
            spacing = (float(thickness) if thickness else 1.0,
                       float(px[0]), float(px[1]))
        except (TypeError, ValueError):
            spacing = None
    else:
        spacing = None

    meta = ScanMetadata(
        modality=modality,
        body_part=str(getattr(ref_ds, "BodyPartExamined", "") or "UNKNOWN").upper(),
        study_uid=getattr(ref_ds, "StudyInstanceUID", None),
        series_uid=series_uid if series_uid != "UNKNOWN_SERIES" else None,
        n_slices=len(slices),
        original_spacing=spacing,
        # Rescale slope/intercept convert stored pixel values -> HU (CT).
        rescale_slope=float(getattr(ref_ds, "RescaleSlope", 1.0) or 1.0),
        rescale_intercept=float(getattr(ref_ds, "RescaleIntercept", 0.0) or 0.0),
        extra={
            "Manufacturer": getattr(ref_ds, "Manufacturer", None),
            "PhotometricInterpretation":
                getattr(ref_ds, "PhotometricInterpretation", None),
        },
    )
    return [p for p, _ in slices], meta


# ---------------------------------------------------------------------------
# Step 2 — modality-specific MONAI preprocessing pipeline
# ---------------------------------------------------------------------------
def build_transform(
    meta: ScanMetadata,
    target_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    ct_window: Optional[Tuple[float, float]] = None,
    is_volumetric: bool = True,
) -> Compose:
    """
    Build a :class:`monai.transforms.Compose` appropriate for ``meta.modality``
    and dimensionality.

    Volumetric (CT/MR): channel-first -> orient RAS -> resample spacing ->
    intensity-normalise -> tensor. Projection 2D (US/XR): channel-first ->
    intensity-normalise -> tensor (no 3D orientation/voxel resampling; add a
    2D ``Resize`` here if you need a fixed input size).
    """
    if is_volumetric:
        head = [
            EnsureChannelFirst(channel_dim="no_channel"),  # (H,W,D)->(1,H,W,D)
            Orientation(axcodes="RAS"),         # deterministic anatomical axes
            Spacing(pixdim=target_spacing[:3], mode="bilinear"),
        ]
    else:
        # 2D projection image: no anatomical orientation / voxel resampling.
        head = [EnsureChannelFirst(channel_dim="no_channel")]  # (H,W)->(1,H,W)

    if meta.modality == Modality.CT:
        # HU windowing: slope/intercept already applied by PydicomReader when
        # available; ScaleIntensityRange then clips to the W/L window -> [0,1].
        width, level = ct_window or CT_WINDOW_PRESETS.get(
            meta.body_part, CT_WINDOW_PRESETS["DEFAULT"]
        )
        a_min, a_max = level - width / 2.0, level + width / 2.0
        intensity = ScaleIntensityRange(
            a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True
        )
    elif meta.modality == Modality.MR:
        # MR intensities are arbitrary units -> robust percentile normalisation
        # is standard (ignores scanner-dependent outliers).
        intensity = ScaleIntensityRangePercentiles(
            lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
        )
    else:  # US / XR / UNKNOWN -> simple robust min-max
        intensity = ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        )

    return Compose(head + [intensity, ToTensor(track_meta=True)])


# ---------------------------------------------------------------------------
# Step 3 — public entry point
# ---------------------------------------------------------------------------
def load_and_preprocess(
    dicom_dir: str | Path,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ct_window: Optional[Tuple[float, float]] = None,
) -> PreprocessedScan:
    """
    End-to-end: directory of DICOM slices -> normalised inference tensor.

    Parameters
    ----------
    dicom_dir : path to a folder containing a DICOM series.
    target_spacing : (z, y, x) mm voxel size to resample to.
    ct_window : optional (width, level) HU override for CT; otherwise chosen
        from ``CT_WINDOW_PRESETS`` by body part.

    Returns
    -------
    PreprocessedScan with a MONAI ``MetaTensor`` (affine preserved) shaped
    ``(C, D, H, W)`` for volumes or ``(C, H, W)`` for single-frame studies.
    """
    file_paths, meta = scan_series(dicom_dir)
    logger.info(
        "Loaded series: modality=%s body_part=%s slices=%d",
        meta.modality.value, meta.body_part, meta.n_slices,
    )

    # PydicomReader assembles pixels + a geometry-correct affine and applies
    # RescaleSlope/Intercept for CT. It groups a *directory* into a series, so
    # we stage only the chosen series' files (via symlinks) into a temp dir —
    # this both isolates the series and avoids the channel-concat path that a
    # bare file list triggers.
    reader = PydicomReader(affine_lps_to_ras=True)
    stage_dir = tempfile.mkdtemp(prefix="dicom_series_")
    try:
        for i, p in enumerate(file_paths):
            link = os.path.join(stage_dir, f"{i:05d}_{os.path.basename(p)}")
            try:
                os.symlink(os.path.abspath(p), link)
            except OSError:
                shutil.copyfile(p, link)  # fallback where symlinks are disallowed
        img_obj = reader.read(stage_dir)
        array, reader_meta = reader.get_data(img_obj)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to assemble pixel volume from {dicom_dir}: {exc}"
        ) from exc
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    # Wrap as a MetaTensor so Spacing/Orientation can use the affine. Cast to
    # float32 up front: torch.quantile (used by percentile normalisation)
    # rejects integer dtypes, and downstream nets expect floats anyway.
    affine = reader_meta.get("affine")
    volume = MetaTensor(
        np.ascontiguousarray(array).astype(np.float32), affine=affine
    )

    # CT/MR are volumetric; US/XR are 2D projections -> different transform head.
    is_volumetric = meta.modality in (Modality.CT, Modality.MR) and array.ndim >= 3
    transform = build_transform(meta, target_spacing, ct_window, is_volumetric)
    try:
        tensor = transform(volume)
    except Exception as exc:
        raise RuntimeError(f"Preprocessing transform failed: {exc}") from exc

    # Record geometry so detections can be mapped back. A0 is the pre-transform
    # (original) affine; A1 is the post-transform (Orientation+Spacing) affine.
    def _np_affine(a):
        return np.asarray(a.detach().cpu()) if hasattr(a, "detach") else np.asarray(a)

    original_affine = _np_affine(affine) if affine is not None else None
    preprocessed_affine = (
        _np_affine(tensor.affine) if hasattr(tensor, "affine") else None
    )

    return PreprocessedScan(
        tensor=tensor.float(),
        metadata=meta,
        file_paths=[str(p) for p in file_paths],
        original_affine=original_affine,
        preprocessed_affine=preprocessed_affine,
        original_shape=tuple(int(s) for s in array.shape),
        preprocessed_shape=tuple(int(s) for s in tensor.shape[1:]),
    )


# ---------------------------------------------------------------------------
# Simple modality router (maps modality/body-part -> registered model key)
# ---------------------------------------------------------------------------
def route_to_model(meta: ScanMetadata) -> str:
    """
    Return a *model registry key* for the given scan. This is intentionally a
    pure string lookup — the actual model objects live in a separate registry
    so ingestion has no heavy weight dependencies.

    Real weights + label taxonomy must be supplied per key before this yields
    clinically meaningful output; unmapped scans fall back to a generic key.
    """
    # Keys match label_taxonomies.MODEL_SPECS. Names encode the true task type
    # (…_seg vs …_classifier) so the post-processor picks the right localizer.
    routing_table = {
        (Modality.CT, "CHEST"):   "ct_chest_nodule_seg",
        (Modality.CT, "LUNG"):    "ct_chest_nodule_seg",
        (Modality.CT, "HEAD"):    "ct_head_ich_classifier",
        (Modality.CT, "BRAIN"):   "ct_head_ich_classifier",
        (Modality.MR, "BRAIN"):   "mr_brain_tumor_seg",
        (Modality.MR, "HEAD"):    "mr_brain_tumor_seg",
        (Modality.US, "BREAST"):  "us_breast_lesion",
        (Modality.XR, "CHEST"):   "cxr_multilabel_classifier",
    }
    key = routing_table.get((meta.modality, meta.body_part))
    if key is None:
        # Fall back on modality-only defaults (generic keys are unmapped in the
        # taxonomy until you register weights + a class map for them).
        key = {
            Modality.CT: "ct_generic_seg",
            Modality.MR: "mr_generic_seg",
            Modality.US: "us_generic_detector",
            Modality.XR: "cxr_multilabel_classifier",
        }.get(meta.modality, "generic_2d_classifier")
        logger.warning("No exact route for %s/%s -> fallback '%s'",
                       meta.modality.value, meta.body_part, key)
    return key


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="DICOM ingestion smoke test")
    parser.add_argument("dicom_dir", help="Path to a DICOM series folder")
    args = parser.parse_args()

    scan = load_and_preprocess(args.dicom_dir)
    print("Modality :", scan.metadata.modality.value)
    print("BodyPart :", scan.metadata.body_part)
    print("Tensor   :", tuple(scan.tensor.shape), scan.tensor.dtype)
    print("Intensity: [%.3f, %.3f]" % (float(scan.tensor.min()),
                                        float(scan.tensor.max())))
    print("Route    :", route_to_model(scan.metadata))
