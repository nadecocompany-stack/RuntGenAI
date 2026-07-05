"""Ingestion, routing, de-identification, and audit tests."""
from __future__ import annotations

import os

import pydicom
import pytest


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------
def test_ingestion_ct_shapes_and_geometry(ct_series):
    from dicom_ingestion import load_and_preprocess, Modality
    scan = load_and_preprocess(ct_series)
    assert scan.tensor.ndim == 4                 # (C, H, W, D)
    assert scan.tensor.shape[0] == 1
    assert scan.metadata.modality is Modality.CT
    assert scan.metadata.body_part == "CHEST"
    # geometry captured for coordinate mapping
    assert scan.original_affine is not None
    assert scan.preprocessed_affine is not None
    assert scan.original_shape is not None


def test_ingestion_normalises_intensity(ct_series):
    from dicom_ingestion import load_and_preprocess
    scan = load_and_preprocess(ct_series)
    t = scan.tensor
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_ingestion_2d_us(us_image):
    from dicom_ingestion import load_and_preprocess, Modality
    scan = load_and_preprocess(us_image)
    assert scan.metadata.modality is Modality.US
    assert scan.tensor.ndim == 3                 # (C, H, W)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def test_route_by_category_all():
    from label_taxonomies import CATEGORY_TO_KEY, route_by_category, get_spec
    for cat, key in CATEGORY_TO_KEY.items():
        assert route_by_category(cat) == key
        assert get_spec(key) is not None


def test_route_by_category_rejects_unknown():
    from label_taxonomies import route_by_category
    with pytest.raises(ValueError):
        route_by_category("not_a_category")


# --------------------------------------------------------------------------
# de-identification
# --------------------------------------------------------------------------
def _inject_phi(directory):
    orig_uids = set()
    for f in sorted(os.listdir(directory)):
        p = os.path.join(directory, f)
        ds = pydicom.dcmread(p, force=True)
        ds.PatientName = "DOE^JANE"
        ds.PatientID = "MRN12345"
        ds.InstitutionName = "General Hospital"
        ds.ReferringPhysicianName = "SMITH^JOHN"
        ds.AccessionNumber = "ACC999"
        ds.add_new(0x00090010, "LO", "PRIVATE_VENDOR")
        orig_uids.add(ds.SeriesInstanceUID)
        ds.save_as(p, enforce_file_format=True)
    return orig_uids


def test_deidentify_removes_phi_and_private(ct_series):
    from deidentify import deidentify_directory, scan_for_phi, DeidConfig
    _inject_phi(ct_series)
    deidentify_directory(ct_series, DeidConfig())
    first = pydicom.dcmread(
        os.path.join(ct_series, sorted(os.listdir(ct_series))[0]), force=True)
    assert scan_for_phi(first) == {}
    assert first.PatientIdentityRemoved == "YES"
    assert (0x0009, 0x0010) not in first          # private tag gone


def test_deidentify_remaps_uids_consistently(ct_series):
    from deidentify import deidentify_directory, DeidConfig
    orig = _inject_phi(ct_series)
    deidentify_directory(ct_series, DeidConfig())
    new = {pydicom.dcmread(os.path.join(ct_series, f), force=True).SeriesInstanceUID
           for f in os.listdir(ct_series)}
    assert len(new) == 1                          # grouping preserved
    assert new.isdisjoint(orig)                   # no longer linkable


def test_deidentify_preserves_pipeline(ct_series):
    from deidentify import deidentify_directory
    from dicom_ingestion import load_and_preprocess
    _inject_phi(ct_series)
    deidentify_directory(ct_series)
    scan = load_and_preprocess(ct_series)         # still ingestible
    assert scan.metadata.modality.value == "CT"


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
def test_audit_chain_intact(tmp_path):
    from audit import AuditLog, sha256_hex
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.record(model_key="m", input_sha256=sha256_hex(bytes([i])),
                   deid_applied=True, confidence=0.5)
    assert len(log) == 3
    assert log.verify() == (True, None)


def test_audit_detects_tampering(tmp_path):
    from audit import AuditLog, sha256_hex
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.record(model_key="m", input_sha256=sha256_hex(bytes([i])))
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace('"m"', '"tampered"')
    path.write_text("\n".join(lines) + "\n")
    ok, bad = log.verify()
    assert ok is False and bad == 2
