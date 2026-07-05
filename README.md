# Radiology Abnormality Detection

Multi-modal DICOM → abnormality **classification + localization + calibrated
confidence** pipeline across CT, MRI, Ultrasound, and X-ray. A DICOM upload is
routed to a modality-specific model, which returns what the abnormality is,
where it is (mapped back to original scan / world coordinates), and how confident
the model is — with the confidence actually calibrated.

> **⚠️ Research prototype — not a medical device, not for clinical use.**
> Models are currently randomly-initialized scaffolding. Every output is a
> placeholder until real weights are trained and externally validated. See
> [Intended use](#intended-use--limitations).

---

## Highlights

- **5 scan categories**, each backed by a specialized model and public benchmark.
- **2 localization regimes**: segmentation → connected-component boxes; and
  weak, Grad-CAM-based boxes for datasets with image-level labels only.
- **Sliding-window inference** so full-resolution CT/MR fit in memory.
- **Calibrated confidence** via temperature scaling (fit + ECE-verified + applied).
- **Coordinate mapping** back to original voxels and patient world-mm.
- **RAD-DINO** (`microsoft/rad-dino`) encoder for the chest-X-ray path.
- **PHI de-identification** + a **tamper-evident audit trail**.
- **PyTorch Lightning training** pipeline (synthetic data now, real-data adapters
  documented).
- **Category-gated React UI** + a **FastAPI** backend.
- **22+ pytest checks**, `mypy` on the pure modules, GitHub Actions CI.

## Categories → datasets → models

| Category | Dataset | Findings | Task | Localization |
|---|---|---|---|---|
| CT · Chest | LIDC-IDRI | Lung nodules | segmentation | mask → boxes |
| CT · Head | RSNA-ICH | 5 hemorrhage subtypes + "any" | classification | Grad-CAM |
| MRI · Brain | BraTS | Whole tumor / core / enhancing | segmentation | mask → boxes |
| Ultrasound · Breast | BUSI | Benign / malignant | detection | mask → boxes |
| X-ray · Chest | CheXpert / MIMIC-CXR | 14 findings | classification | Grad-CAM |

## Repository layout

```
radiology_platform/
├── README.md · ARCHITECTURE.md · CLAUDE.md · CONTRIBUTING.md · GITHUB_SETUP.md
├── LICENSE · .gitignore · .gitattributes
├── requirements.txt · pyproject.toml
├── index.html                     # self-contained walkthrough + document hub
├── .github/workflows/ci.yml       # CI: mypy + pytest (3.10 / 3.11)
├── src/
│   ├── dicom_ingestion.py         # DICOM → tensor; series grouping, geometry, router
│   ├── label_taxonomies.py        # per-model taxonomy + category routing
│   ├── model_registry.py          # builds each architecture; loads weights
│   ├── inference.py               # sliding-window volumetric inference (+ AMP)
│   ├── confidence_localization.py # seg logits → calibrated confidence + boxes
│   ├── explainability.py          # Grad-CAM → boxes (CNN + ViT)
│   ├── calibration.py             # temperature scaling: fit + ECE + apply + store
│   ├── coordinate_mapping.py      # boxes → original voxel + world mm
│   ├── deidentify.py              # DICOM PHI removal + UID remapping
│   ├── audit.py                   # tamper-evident hash-chained inference log
│   ├── cxr_rad_dino.py            # microsoft/rad-dino CXR encoder + head + Grad-CAM
│   ├── api.py                     # FastAPI: category routing + calibrated /analyze
│   └── training/                  # Lightning training pipeline
│       ├── synthetic_dataset.py · datamodule.py · lit_module.py · train.py
├── tests/                         # canonical pytest suite
├── scripts/                       # runnable demos (mirror the tests, with output)
└── ui/RadiologyIntake.jsx         # category-gated upload UI (React)
```

## Setup

```bash
pip install -r requirements.txt
```

Runs on CPU; a CUDA build of PyTorch is used automatically if present.

## Quickstart

```bash
# end-to-end plumbing across all 5 modalities (synthetic DICOM, no data needed)
python scripts/run_smoke_test.py

# training demo: a segmentation model + a classifier learn on synthetic data
python scripts/train_demo.py

# API (category-forced routing, calibrated + coordinate-mapped output)
cd src && uvicorn api:app --reload
#   GET  /categories   -> selectable categories + findings per model
#   POST /analyze      -> multipart: category + file(s) -> detections + confidence
```

The UI (`ui/RadiologyIntake.jsx`, React) runs standalone with simulated
inference, or point it at `/analyze` for live results.

## How it works

Eight stages per scan: **intake & category → ingestion (DICOM→tensor, PHI
de-identified) → routing → model + sliding-window inference → localization →
calibration → coordinate mapping → output** (JSON: top finding, per-detection
boxes in model / original-voxel / world-mm space, calibrated confidence,
temperature, plus a tamper-evident audit record). Open `index.html` for the full
visual walkthrough.

## Training

One Lightning pipeline trains any model, dispatching loss/metric by task type
(DiceCE + Dice for segmentation, BCE/CE + accuracy for classification), then fits
and stores the calibration temperature and saves a `state_dict` that
`ModelRegistry.load_weights` consumes.

```bash
python -m training.train ct_chest_nodule_seg --epochs 8   # from src/
```

On synthetic data the demo reaches foreground Dice ~0.02 → ~0.91 (segmentation)
and reduces classification loss + ECE — proving the loop *learns*, not just
executes. **Using real data:** implement a `Dataset` yielding the same batch dict
(`{"image", "label"}`, shapes documented in `training/datamodule.py`) and pass it
as `train_ds` / `val_ds`. `datamodule.DATA_REQUIREMENTS` lists the source, format,
and adapter notes for each dataset.

## Confidence calibration

Raw softmax/sigmoid outputs are over-confident; temperature scaling (one scalar
fit on validation data) corrects this and is applied at inference. Fit once,
store, and the API applies it automatically:

```python
from calibration import calibrate_and_store, CalibrationStore
store = CalibrationStore("calibration.json")
calibrate_and_store("cxr_multilabel_classifier", val_logits, val_labels,
                    mode="multilabel", store=store)   # reports ECE before/after
```

## Privacy & audit

- **De-identification** (`deidentify.py`) blanks direct identifiers, removes
  private tags, and remaps UIDs consistently (grouping preserved, no longer
  linkable). Pipeline-critical tags are never touched. On by default in the API
  (`DEIDENTIFY=0` to skip). It does **not** catch burned-in pixel PHI.
- **Audit trail** (`audit.py`) appends one hash-chained JSONL record per
  inference (model + code version, a PHI-free content hash of the processed
  pixels, de-id status, output summary). `AuditLog.verify()` detects tampering.

## Testing & CI

```bash
pip install pytest mypy
pytest        # ingestion, routing, de-id, audit, mapping, calibration, inference, training
mypy          # type-checks the pure-logic modules
```

CI runs both on Python 3.10 and 3.11 (`.github/workflows/ci.yml`). `tests/` is the
source of truth; `scripts/` are runnable demos with printed output.

## Intended use & limitations

- **Not a medical device.** Not cleared or validated for diagnosis, triage, or
  any clinical decision.
- **Un-trained weights.** Numbers shown anywhere are placeholders.
- **Calibration ≠ accuracy.** Temperature scaling fixes confidence honesty, not
  discrimination, and assumes validation data matches deployment.
- **Human in the loop.** Real deployment requires trained + externally validated
  models, regulatory review, robust PHI handling, and a qualified reader.

## License

MIT — see [`LICENSE`](LICENSE) (replace the placeholder copyright holder). This
software is a research prototype and is not a medical device.
