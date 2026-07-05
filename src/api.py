"""
api.py
==================================================================
Thin HTTP layer over the pipeline. The key idea: the user picks a CATEGORY in
the UI, and that choice FORCES routing to the correct specialized model —
overriding DICOM metadata, which is often missing or wrong. This is what lets
the platform apply modality-specific knowledge to actually find anomalies.

Endpoints
---------
GET  /categories  -> taxonomy for the UI (findings, task, localization type)
POST /analyze     -> multipart: category + file(s); returns detections + conf

Run:  uvicorn api:app --reload   (from src/)

NOTE: models are randomly initialized until real weights are registered, so
detection payloads are placeholders. The routing, ingestion, and response
contract are real.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import List

import torch

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install API deps: pip install 'fastapi[standard]' uvicorn") from exc

from dicom_ingestion import load_and_preprocess, ScanMetadata, Modality
from label_taxonomies import (
    CATEGORY_TO_KEY, route_by_category, get_spec, Localization,
)
from model_registry import ModelRegistry, _in_channels
from confidence_localization import ConfidenceLocalizer
from explainability import compute_gradcam, heatmap_to_boxes
from calibration import CalibrationStore, apply_temperature
from deidentify import deidentify_directory, DeidConfig
from audit import AuditLog, sha256_hex

app = FastAPI(title="Radiology Abnormality Detection", version="0.1.0")
PLATFORM_VERSION = "0.1.0"
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_REGISTRY = ModelRegistry(device="cpu", seed=0)
# Per-model temperatures, persisted if a path is set. Missing keys -> T=1.0.
_CALIB = CalibrationStore(os.environ.get("CALIBRATION_JSON"))
# PHI de-identification is ON by default; set DEIDENTIFY=0 to disable (e.g. when
# uploads are already de-identified upstream).
_DEID = os.environ.get("DEIDENTIFY", "1") != "0"
# Tamper-evident audit trail of every inference (provenance, never PHI).
_AUDIT = AuditLog(os.environ.get("AUDIT_LOG", "audit.jsonl"))
DISCLAIMER = "Research prototype. Not a medical device. Not for clinical use."


@app.get("/categories")
def categories() -> dict:
    """Return the selectable categories and what each model looks for."""
    out = []
    for cat, key in CATEGORY_TO_KEY.items():
        spec = get_spec(key)
        out.append({
            "category": cat,
            "model_key": key,
            "dataset": spec.dataset,
            "modality": spec.modality,
            "body_part": spec.body_part,
            "task": spec.task.value,
            "localization": spec.localization.value,
            "findings": [v for i, v in sorted(spec.class_map.items()) if i != 0
                         or spec.localization is Localization.GRADCAM],
        })
    return {"categories": out, "disclaimer": DISCLAIMER}


def _stage_upload(files: List[UploadFile]) -> str:
    """Save uploads into a temp dir; expand a single zip into a series."""
    d = tempfile.mkdtemp(prefix="upload_")
    for f in files:
        dest = os.path.join(d, os.path.basename(f.filename or "scan.dcm"))
        with open(dest, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        if dest.lower().endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                z.extractall(d)
            os.remove(dest)
    return d


@app.post("/analyze")
async def analyze(
    category: str = Form(...),
    files: List[UploadFile] = File(...),
) -> dict:
    # 1) Category forces the model — independent of DICOM metadata.
    try:
        key = route_by_category(category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    spec = get_spec(key)

    # 2) De-identify PHI, then ingest the uploaded series.
    work = _stage_upload(files)
    deid_applied = False
    if _DEID:
        try:
            deidentify_directory(work, DeidConfig())
            deid_applied = True
        except Exception as exc:   # never proceed with un-scrubbed PHI silently
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"De-identification failed: {exc}")
    try:
        scan = load_and_preprocess(work)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read scan: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Provenance hash of the PROCESSED pixel data (PHI-free, reproducible).
    input_sha256 = sha256_hex(scan.tensor.detach().cpu().numpy().tobytes())

    # 3) Build model + shape input, then infer + localize per task type.
    model = _REGISTRY.build(key)
    x = scan.tensor
    want = _in_channels(spec)
    if x.shape[0] != want:                      # e.g. BraTS expects 4 sequences
        x = x[:1].repeat(want, *([1] * (x.ndim - 1)))
    x = x.unsqueeze(0)

    detections = []
    temperature = _CALIB.get(key)   # 1.0 until a validation-fit T is stored
    geom = dict(
        preprocessed_affine=scan.preprocessed_affine,
        original_affine=scan.original_affine,
        clip_shape=scan.original_shape,
    )
    if spec.localization is Localization.MASK_CC:
        from coordinate_mapping import map_detection_bbox
        from inference import segment_volume
        # Sliding-window inference: fits arbitrarily large volumes and returns
        # full-resolution logits in preprocessed-voxel space (no pad offset).
        logits = segment_volume(model, x, overlap=0.25, mode="gaussian")[0]
        loc = ConfidenceLocalizer(
            mode="sigmoid" if spec.multilabel else "softmax",
            prob_threshold=0.5, min_region_voxels=10,
            temperature=temperature,          # calibrated confidence
        )
        result = loc(logits)
        top_class, top_conf = result.top_class, result.top_confidence
        for d in result.detections[:20]:
            mapped = map_detection_bbox(d.bbox, **geom)
            detections.append({
                "class": spec.class_map.get(d.class_id, str(d.class_id)),
                "confidence": round(d.confidence, 3),
                "bbox_model": list(d.bbox),
                "bbox_original_voxel": list(mapped.get("original_voxel", [])),
                "bbox_world_mm": list(mapped.get("world_mm", [])),
            })
    else:  # classification + Grad-CAM
        from coordinate_mapping import map_detection_bbox
        pre_shape = tuple(int(s) for s in x.shape[2:])
        logits = model(x)
        mode = "multilabel" if spec.multilabel else "multiclass"
        probs = apply_temperature(logits, temperature, mode)[0]  # calibrated
        scores = probs.clone(); scores[0] = -1
        top_class = int(scores.argmax()); top_conf = float(probs[top_class])
        cam = compute_gradcam(model, x, model.feature_layer, top_class,
                              multilabel=spec.multilabel)
        for b in heatmap_to_boxes(cam, top_class, min_region_voxels=2)[:20]:
            mapped = map_detection_bbox(
                b.bbox, feature_shape=cam.shape, input_shape=pre_shape, **geom)
            detections.append({
                "class": spec.class_map.get(top_class, str(top_class)),
                "confidence": round(b.peak_confidence, 3),
                "bbox_model": list(b.bbox),
                "bbox_original_voxel": list(mapped.get("original_voxel", [])),
                "bbox_world_mm": list(mapped.get("world_mm", [])),
            })

    # Append a tamper-evident audit record (provenance, no PHI).
    audit_entry = _AUDIT.record(
        model_key=key,
        code_version=PLATFORM_VERSION,
        input_sha256=input_sha256,
        deid_applied=deid_applied,
        temperature=round(float(temperature), 4),
        top_finding=spec.class_map.get(top_class, "n/a") if top_class is not None else "n/a",
        confidence=round(float(top_conf), 3),
        n_detections=len(detections),
    )

    return {
        "model_key": key,
        "modality": scan.metadata.modality.value,
        "body_part": scan.metadata.body_part,
        "task": spec.task.value,
        "temperature": round(float(temperature), 4),
        "top_finding": spec.class_map.get(top_class, "n/a") if top_class is not None else "n/a",
        "confidence": round(float(top_conf), 3),
        "detections": detections,
        "deidentified": deid_applied,
        "audit": {"event_id": audit_entry["event_id"],
                  "input_sha256": input_sha256,
                  "record_hash": audit_entry["record_hash"]},
        "disclaimer": DISCLAIMER,
    }
