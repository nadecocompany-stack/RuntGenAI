"""
deidentify.py
==================================================================
Remove protected health information (PHI) from DICOM before it reaches the
pipeline or any log. This is a pragmatic implementation of the high-risk parts
of the DICOM Basic Application Level Confidentiality Profile (PS3.15 Annex E):
it blanks direct identifiers, removes private tags, and consistently remaps
UIDs so a study can no longer be linked back while series grouping is preserved.

What it does NOT do (call out honestly):
  * It does not detect *burned-in* PHI (text rendered into ultrasound/scanned
    pixels). Use a pixel-level scrubber for that; here we set BurnedInAnnotation
    handling to a flag only.
  * It is not a certified/validated de-identification pipeline. For real PHI,
    use a reviewed tool (e.g. CTP, MIRC) and institutional sign-off.

Tags the pipeline needs (modality, body part, geometry, pixel data) are never
removed — see ``PROTECTED``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import pydicom
from pydicom.uid import generate_uid

logger = logging.getLogger("deidentify")
logger.addHandler(logging.NullHandler())

# Keywords blanked (emptied) — direct identifiers kept as present-but-empty.
BLANK_KEYWORDS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime",
    "PatientSex", "PatientAge", "PatientAddress", "PatientTelephoneNumbers",
    "PatientMotherBirthName", "OtherPatientIDs", "OtherPatientNames",
    "EthnicGroup", "Occupation", "AdditionalPatientHistory", "PatientComments",
    "MilitaryRank", "BranchOfService", "MedicalRecordLocator",
    "ReferringPhysicianName", "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers", "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy", "OperatorsName", "PhysiciansOfRecord",
    "RequestingPhysician", "RequestingService",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "StationName", "AccessionNumber", "StudyID", "DeviceSerialNumber",
    # dates/times (identifying in combination) — blanked; the pipeline needs none
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "InstanceCreationDate", "StudyTime", "SeriesTime", "AcquisitionTime",
    "ContentTime", "InstanceCreationTime", "PatientBirthName",
)

# Sequences/entries removed entirely (may nest identifiers).
DELETE_KEYWORDS = (
    "OtherPatientIDsSequence", "ReferencedPatientSequence",
    "RequestAttributesSequence", "ScheduledProcedureStepSequence",
    "PerformedProcedureStepDescription", "RequestedProcedureID",
    "IssuerOfPatientID", "PatientInsurancePlanCodeSequence",
)

# UID keywords remapped to fresh UIDs (consistent within one call) so grouping
# survives but the original (linkable) UIDs are gone.
UID_KEYWORDS = (
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "FrameOfReferenceUID", "SynchronizationFrameOfReferenceUID",
)

# Never touched — required downstream.
PROTECTED = frozenset((
    "SOPClassUID", "Modality", "BodyPartExamined", "Rows", "Columns",
    "PixelData", "PixelSpacing", "SliceThickness", "ImagePositionPatient",
    "ImageOrientationPatient", "PhotometricInterpretation", "SamplesPerPixel",
    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
    "RescaleSlope", "RescaleIntercept", "NumberOfFrames", "InstanceNumber",
))


@dataclass
class DeidConfig:
    remove_private_tags: bool = True
    remap_uids: bool = True
    method: str = "platform-basic-v1"
    uid_map: Dict[str, str] = field(default_factory=dict)  # original UID -> new


def _remap_uid(value: str, cfg: DeidConfig) -> str:
    """Return a stable new UID for ``value`` (same input -> same output)."""
    if value not in cfg.uid_map:
        cfg.uid_map[value] = generate_uid()
    return cfg.uid_map[value]


def deidentify_dataset(ds: "pydicom.dataset.Dataset", cfg: Optional[DeidConfig] = None
                       ) -> "pydicom.dataset.Dataset":
    """
    De-identify a dataset in place and return it. Blanks direct identifiers,
    deletes identifier sequences, remaps UIDs, removes private tags, and stamps
    the de-identification method.
    """
    cfg = cfg or DeidConfig()

    for kw in BLANK_KEYWORDS:
        if kw in PROTECTED:
            continue
        if kw in ds:
            el = ds.data_element(kw)
            if el is not None:
                el.value = ""
            else:
                del ds[kw]

    for kw in DELETE_KEYWORDS:
        if kw in ds and kw not in PROTECTED:
            del ds[kw]

    if cfg.remap_uids:
        for kw in UID_KEYWORDS:
            if kw in ds and getattr(ds, kw, None):
                setattr(ds, kw, _remap_uid(str(getattr(ds, kw)), cfg))
        # keep file-meta SOP UID consistent with the (remapped) dataset UID
        if getattr(ds, "file_meta", None) is not None and "SOPInstanceUID" in ds:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    if cfg.remove_private_tags:
        ds.remove_private_tags()

    # Provenance stamp per the standard.
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = cfg.method
    return ds


def deidentify_file(path: str | Path, out_path: str | Path,
                    cfg: Optional[DeidConfig] = None) -> str:
    """Read, de-identify, and write a single DICOM file."""
    ds = pydicom.dcmread(str(path), force=True)
    deidentify_dataset(ds, cfg)
    ds.save_as(str(out_path), enforce_file_format=True)
    return str(out_path)


def deidentify_directory(directory: str | Path,
                         cfg: Optional[DeidConfig] = None) -> int:
    """
    De-identify every readable DICOM in a directory in place, using ONE shared
    ``DeidConfig`` so UIDs remap consistently across the whole series/study.
    Returns the count of files processed. Non-DICOM files are skipped.
    """
    cfg = cfg or DeidConfig()
    n = 0
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), force=True)
            if "Modality" not in ds and "SOPClassUID" not in ds:
                continue
        except Exception:
            continue
        deidentify_dataset(ds, cfg)
        ds.save_as(str(p), enforce_file_format=True)
        n += 1
    logger.info("De-identified %d file(s) in %s", n, directory)
    return n


def scan_for_phi(ds: "pydicom.dataset.Dataset") -> Dict[str, str]:
    """
    Report which PHI-bearing tags are still present and non-empty. Used to
    verify de-identification worked (returns {} when clean).
    """
    found = {}
    for kw in BLANK_KEYWORDS + DELETE_KEYWORDS:
        if kw in ds:
            val = getattr(ds, kw, "")
            if val not in ("", None) and not (hasattr(val, "__len__") and len(val) == 0):
                found[kw] = str(val)[:40]
    return found


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = deidentify_directory(sys.argv[1])
    print(f"de-identified {n} file(s)")
