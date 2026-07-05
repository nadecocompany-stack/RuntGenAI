"""
synthetic_dicom.py
==================================================================
Generate small, valid, UNCOMPRESSED DICOM series for each modality so the
smoke test can exercise the real ingestion path (pydicom + MONAI) without any
patient data. These images are pure noise + a planted bright blob — they are
NOT medical data and carry no diagnostic meaning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    generate_uid,
    ExplicitVRLittleEndian,
    CTImageStorage,
    MRImageStorage,
    UltrasoundImageStorage,
    ComputedRadiographyImageStorage,
)

_SOP_BY_MODALITY = {
    "CT": CTImageStorage,
    "MR": MRImageStorage,
    "US": UltrasoundImageStorage,
    "CR": ComputedRadiographyImageStorage,  # projection X-ray
}


def _base_dataset(modality: str, body_part: str, series_uid: str,
                  study_uid: str, rows: int, cols: int) -> Dataset:
    sop_class = _SOP_BY_MODALITY[modality]
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    return ds


def _blob_image(rows: int, cols: int, base: float, span: float) -> np.ndarray:
    img = (np.random.rand(rows, cols) * span + base)
    r0, c0 = rows // 3, cols // 3
    img[r0:r0 + rows // 4, c0:c0 + cols // 4] += span * 0.8  # planted bright blob
    return img.astype(np.uint16)


def make_series(
    out_dir: str | Path,
    modality: str,
    body_part: str,
    volumetric: bool,
    n_slices: int = 8,
    shape: Tuple[int, int] = (64, 64),
    pixel_spacing: Tuple[float, float] = (0.7, 0.7),
    slice_thickness: float = 1.0,
    ct_rescale: Tuple[float, float] = (1.0, -1024.0),
) -> str:
    """Write one synthetic series into ``out_dir`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = shape
    series_uid, study_uid = generate_uid(), generate_uid()
    n = n_slices if volumetric else 1

    for i in range(n):
        ds = _base_dataset(modality, body_part, series_uid, study_uid, rows, cols)
        ds.InstanceNumber = i + 1
        ds.PixelSpacing = [pixel_spacing[0], pixel_spacing[1]]
        if modality == "CT":
            ds.RescaleSlope, ds.RescaleIntercept = ct_rescale
            base, span = 1024.0, 400.0     # stored values -> HU via rescale
        elif modality == "MR":
            base, span = 100.0, 500.0
        else:                              # US / CR
            base, span = 0.0, 3000.0
        if volumetric:
            ds.SliceThickness = slice_thickness
            ds.ImagePositionPatient = [0.0, 0.0, float(i) * slice_thickness]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelData = _blob_image(rows, cols, base, span).tobytes()
        ds.save_as(out_dir / f"slice_{i:03d}.dcm", enforce_file_format=True)

    return str(out_dir)


def generate_all(base_dir: str | Path, seed: int = 0) -> Dict[str, str]:
    """
    Create one series per routing case. Returns {case_name: directory}.
    CR modality is used for chest X-ray (folds into the XR route).
    """
    np.random.seed(seed)
    base_dir = Path(base_dir)
    cases = {
        "ct_chest": dict(modality="CT", body_part="CHEST", volumetric=True),
        "ct_head":  dict(modality="CT", body_part="HEAD", volumetric=True),
        "mr_brain": dict(modality="MR", body_part="BRAIN", volumetric=True),
        "us_breast": dict(modality="US", body_part="BREAST", volumetric=False,
                          shape=(128, 128)),
        "xr_chest": dict(modality="CR", body_part="CHEST", volumetric=False,
                         shape=(128, 128)),
    }
    return {name: make_series(base_dir / name, **kwargs)
            for name, kwargs in cases.items()}


if __name__ == "__main__":  # pragma: no cover
    import tempfile
    dirs = generate_all(tempfile.mkdtemp(prefix="synthetic_dicom_"))
    for name, path in dirs.items():
        print(f"{name:10s} -> {path} ({len(os.listdir(path))} files)")
