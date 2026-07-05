# Radiology Abnormality Detection Platform — Architecture

> **Scope guard.** This system is an *engineering + computer-vision* pipeline.
> Every recommended model below (including RAD-DINO) is research-grade and
> **not cleared for clinical use**. Outputs are probabilities and coordinates,
> not diagnoses. Any deployment needs regulatory review, clinical validation,
> and a human-in-the-loop.

## 1. High-level flow

```
DICOM upload
   │
   ▼
┌──────────────────────┐   pydicom header parse (Modality, BodyPartExamined,
│  Ingestion / Router  │   PixelSpacing, RescaleSlope/Intercept)
└──────────┬───────────┘
           │  ScanMetadata + normalized tensor  (dicom_ingestion.py)
           ▼
┌──────────────────────┐   route_to_model(metadata) -> registry key
│   Model Registry     │   (CT/MR/US/XR × body part -> specialized sub-model)
└──────────┬───────────┘
           ▼
┌──────────────────────┐   segmentation / detection / classification head
│  Specialized model   │   emits raw logits
└──────────┬───────────┘
           ▼
┌──────────────────────┐   softmax/sigmoid + temperature scaling -> confidence
│ Confidence & Localize│   connected components -> bounding boxes
└──────────┬───────────┘   (confidence_localization.py)
           ▼
┌──────────────────────┐   Grad-CAM / Grad-CAM++ heatmap over the input
│   Explainability     │   (pytorch-grad-cam)
└──────────┬───────────┘
           ▼
  {classification, bbox/mask, calibrated confidence, heatmap}
```

## 2. Routing mechanism

Routing is a **pure metadata lookup**, kept free of heavy model imports so the
ingestion service stays lightweight and horizontally scalable.

1. `scan_series()` walks the upload dir, tolerates corrupted/non-image files,
   groups slices by `SeriesInstanceUID`, and keeps the largest coherent series.
2. `Modality` (0008,0060) is normalized — DICOM emits `CR`/`DX`/`RF` for
   projection X-ray, which we fold into a single `XR` token; `MRI→MR`, etc.
3. `route_to_model(metadata)` maps `(modality, body_part)` to a **registry key**
   (a string). The registry (which you populate with weights) resolves the key
   to an actual model. Unmapped scans fall back to modality-only defaults.

Keeping specialized sub-models beats one universal model because intensity
semantics differ fundamentally across modalities (HU for CT vs arbitrary units
for MR vs echo intensity for US), and lesion appearance is organ-specific.

## 3. Recommended model architectures per modality

| Modality / region | Recommended architecture | Output | Rationale |
|---|---|---|---|
| **CT (volumetric)** | **3D U-Net** (`monai.networks.nets.UNet`) or **SegResNet**; **UNETR/SwinUNETR** if you want a transformer encoder | Voxel segmentation mask | 3D U-Net is the de-facto standard for volumetric medical segmentation; captures cross-slice context a 2D model misses. |
| **MR (volumetric)** | **UNETR / SwinUNETR** (`monai.networks.nets`) | Voxel segmentation mask | Transformer encoders handle the larger contextual/contrast variation in MR well; drop-in with MONAI. |
| **Ultrasound (2D/2D+t)** | **Mask R-CNN** or **RetinaNet** (`torchvision.models.detection`); MONAI also ships a `RetinaNetDetector` | Bounding boxes (+ instance masks) | US abnormalities are usually localized objects, not dense volumetric structures — an object detector fits better than segmentation. |
| **Chest X-ray (2D)** | **RAD-DINO** (`microsoft/rad-dino`, DINOv2 ViT-B, 768-d) as a frozen encoder + a lightweight classification/segmentation head | Class scores; boxes via a detection head or via Grad-CAM localization | RAD-DINO is a strong self-supervised CXR *encoder* trained on ~880k CXRs; per Microsoft it's a backbone you attach heads to, **research-only**. Not a standalone detector. |

Notes:
- **Detection vs segmentation.** Use segmentation (masks) where "where to look"
  means a region/organ boundary (CT/MR); use detection (boxes) where it means a
  discrete finding (US, and CXR if you train a detection head).
- **ViTs for classification** (as in your plan) are supported here through
  RAD-DINO on CXR and UNETR/SwinUNETR encoders on volumes.

## 4. Confidence & explainability

- **Confidence** = softmax (mutually-exclusive classes) or sigmoid (multi-label)
  over logits, with **temperature scaling** (Guo et al., 2017) actually applied
  at inference via a per-model `CalibrationStore`. `calibration.py` fits T on
  validation data (multiclass *or* multilabel), measures Expected Calibration
  Error before/after, and persists T; unfitted models use T=1.0 (no-op). See
  `ConfidenceLocalizer.fit_temperature`. — superseded by `calibration.py`.
- **Localization** thresholds the probability map, runs
  `scipy.ndimage.label` for connected components, and emits a bounding box +
  mean/peak confidence + size per region (2D and 3D).
- **Explainability** — Grad-CAM heatmaps show which input regions drove a
  prediction. `explainability.compute_gradcam` is dependency-free and handles
  both CNNs (conv activations) and **ViT backbones** via a `reshape_transform`
  that folds the token sequence back to a patch grid. For **RAD-DINO** the
  target layer is the last block's `norm1`; the CLS prefix token is dropped and
  the 37×37 patch grid is reconstructed (`cxr_rad_dino.py`). Grad-CAM over a
  frozen encoder requires the input tensor to carry grad — handled internally.

## 5. Module map

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies. |
| `src/dicom_ingestion.py` | DICOM → metadata + normalized tensor; modality router. |
| `src/confidence_localization.py` | Segmentation logits → calibrated confidence + bounding boxes. |
| `src/inference.py` | Sliding-window volumetric inference (fits large CT/MR) + AMP. |
| `src/coordinate_mapping.py` | Detection boxes → original voxel + world-mm coordinates. |
| `src/label_taxonomies.py` | Per-model dataset, task type, class map, localization; category routing. |
| `src/explainability.py` | Grad-CAM saliency → boxes for the classification models. |
| *(to add)* `src/model_registry.py` | Maps registry keys → loaded model objects + weights. |

## 6. Datasets, taxonomies & the two localization regimes

Registry keys (from `route_to_model`) map to datasets in `label_taxonomies.py`:

| Key | Dataset | Modality/region | Task | Localization |
|---|---|---|---|---|
| `ct_chest_nodule_seg` | LIDC-IDRI | CT chest | segmentation | supervised mask → CC |
| `ct_head_ich_classifier` | RSNA ICH | CT head | classification | **Grad-CAM** → boxes |
| `mr_brain_tumor_seg` | BraTS | MR brain | segmentation | supervised mask → CC |
| `us_breast_lesion` | BUSI | US breast | detection | supervised mask → CC |
| `cxr_multilabel_classifier` | CheXpert / MIMIC-CXR | X-ray chest | classification | **Grad-CAM** → boxes |

**This is the key design consequence:** LIDC, BraTS, and BUSI ship pixel/voxel
annotations, so `ConfidenceLocalizer` produces boxes directly from predicted
masks. RSNA-ICH and CheXpert/MIMIC-CXR ship **only image/slice-level labels** —
they cannot supervise a box. For those two, `explainability.heatmap_to_boxes`
derives boxes from a Grad-CAM saliency map instead. Both paths emit the same
`Detection` schema, so downstream code is uniform.

**Dataset-specific gotchas encoded in the taxonomy:**
- **BraTS** raw voxel labels are `{1: NCR/NET, 2: edema, 4: enhancing}` (label 3
  unused) for ≤2021/MSD releases; **BraTS 2023+ relabeled enhancing 4→3**.
  Models are scored on nested regions WT/TC/ET. Confirm your release's integers.
- **RSNA-ICH** is multi-label (5 subtypes + "any"), 3-window slice input; no masks.
- **CheXpert** has 14 observations with uncertainty labels `{0,1,-1,blank}` —
  pick a policy (U-Ones/U-Zeros/ignore) per class when building targets.
- **BUSI** "normal" images carry empty masks; benign/malignant carry lesion masks.
- **LIDC-IDRI** has up to 4 readers per nodule; build consensus masks (e.g. ≥50%
  agreement via `pylidc`) and optionally a 1–5 malignancy head.

## 7. What is still needed to go from scaffold → real inference

1. **Trained weights** per registry key (or a plan to train them). None of the
   recommended nets ship abnormality-detection weights for arbitrary organs.
   *(Label taxonomies are now defined in `label_taxonomies.py`.)*
2. **`model_registry.py`** to instantiate each architecture with the correct
   output-channel count (`ModelSpec.num_classes`) and load the weights above.
   This is the one remaining scaffold file; it needs the weights to be useful.
3. *(Optional)* Your preferred **CT window presets** per body part if they
   should differ from the defaults in `CT_WINDOW_PRESETS`.
