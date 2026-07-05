# CLAUDE.md

Guidance for Claude Code when working in this repository. Claude reads this file
automatically on every invocation.

## What this is

A multi-modal radiology **abnormality-detection pipeline** (research/engineering
scaffold). It ingests DICOM, routes by category to a specialized model, and
returns classification + localization + calibrated confidence, with detections
mapped back to original scan / world coordinates.

> **Not a medical device. Not for clinical use.** Models are currently
> randomly-initialized scaffolding. Any output is a placeholder until real
> weights are trained and externally validated.

## Hard rules (do not violate)

- **Never commit** real DICOM, datasets, PHI, or trained weights. `.gitignore`
  blocks `*.dcm`, `*.nii*`, `data/`, `*.pt`, `*.ckpt`, `calibration.json`,
  `audit.jsonl`. Keep it that way.
- **Preserve the "not for clinical use" disclaimers** in README, ARCHITECTURE,
  index.html, and API responses.
- **De-identification and audit** (`src/deidentify.py`, `src/audit.py`) are
  safety-relevant — do not weaken them without explicit instruction.

## Layout

- `src/` — pipeline modules (ingestion, routing, registry, inference,
  localization, calibration, coordinate mapping, de-id, audit, RAD-DINO, api).
- `src/training/` — PyTorch Lightning training pipeline.
- `tests/` — canonical pytest suite (the source of truth).
- `scripts/` — runnable demos (smoke test, rad-dino, coordinate mapping,
  sliding-window, calibration, training) — mirror the tests with printed output.
- `ui/RadiologyIntake.jsx` — category-gated React upload UI.
- `index.html` — self-contained walkthrough + document hub.

## Commands

```bash
pip install -r requirements.txt          # deps
pytest                                   # run the test suite (canonical)
mypy                                     # type-check pure modules
python scripts/run_smoke_test.py         # end-to-end plumbing across 5 modalities
python scripts/train_demo.py             # training demo (seg + classifier)
cd src && uvicorn api:app --reload       # API
```

Always run `pytest` and `mypy` before opening a PR; CI runs both on 3.10/3.11.

## Conventions

- Model behaviour is driven by `label_taxonomies.ModelSpec` — task type
  (segmentation vs classification), `multilabel`, `class_map`, and localization
  strategy. New models are added there + in `model_registry.py`.
- **Two localization regimes**: mask→connected-components for segmentation;
  Grad-CAM→boxes for classification (image-level labels only).
- **Coordinate spaces** matter: localizers emit boxes in feature/preprocessed
  space; `coordinate_mapping.py` maps them to original voxel + world mm via the
  MetaTensor affines. Don't return raw feature-grid coordinates to users.
- **Calibration**: confidence uses temperature scaling (`calibration.py`);
  unfitted models default to T=1.0.

## Style

Prefer small, single-responsibility modules; keep ingestion free of model deps
and the registry free of ingestion deps. Match the existing docstring style
(module purpose + honest caveats).
