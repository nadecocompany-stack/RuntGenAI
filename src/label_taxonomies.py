"""
label_taxonomies.py
==================================================================
Ground-truth mapping from each routed model key to its dataset, task type,
class taxonomy, and localization strategy.

WHY THIS MATTERS
----------------
Three of the target datasets supervise *pixel/voxel localization* directly
(LIDC-IDRI, BraTS, BUSI -> masks/contours). Two do NOT (RSNA-ICH and
CheXpert/MIMIC-CXR -> image/slice-level labels only). For the latter, a
bounding box can only be produced *weakly*, by thresholding a Grad-CAM
saliency map. The ``localization`` field records which path applies so the
inference layer picks the right post-processor:

  * TaskType.SEGMENTATION -> ConfidenceLocalizer(segmentation logits)
  * TaskType.CLASSIFICATION -> GradCAMLocalizer(model, target layer)

All class maps below are transcribed from the public dataset definitions and
annotated with source notes. Verify against your downloaded release — BraTS in
particular changed its enhancing-tumor integer label in 2023 (see note).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TaskType(str, Enum):
    SEGMENTATION = "segmentation"      # dense voxel/pixel masks
    DETECTION = "detection"            # bounding boxes / instance masks
    CLASSIFICATION = "classification"  # image/slice-level labels only


class Localization(str, Enum):
    MASK_CC = "mask_connected_components"   # from predicted mask (supervised)
    GRADCAM = "gradcam_derived_boxes"       # from saliency map (weakly-sup.)


@dataclass
class ModelSpec:
    """Everything the registry + post-processor need for one routed model."""
    key: str
    dataset: str
    modality: str
    body_part: str
    task: TaskType
    localization: Localization
    # index -> human-readable finding. Index 0 is background for segmentation.
    class_map: Dict[int, str]
    recommended_arch: str
    # For sigmoid multi-label classification vs softmax mutually-exclusive.
    multilabel: bool = False
    notes: str = ""
    modalities_required: List[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_map)


# ---------------------------------------------------------------------------
# CT / Chest — LIDC-IDRI (nodule contours available -> supervised localization)
# ---------------------------------------------------------------------------
CT_CHEST_NODULE = ModelSpec(
    key="ct_chest_nodule_seg",
    dataset="LIDC-IDRI",
    modality="CT", body_part="CHEST",
    task=TaskType.SEGMENTATION,
    localization=Localization.MASK_CC,
    class_map={0: "background", 1: "nodule"},
    recommended_arch="3D U-Net / SegResNet (monai.networks.nets)",
    notes=("LIDC provides per-radiologist nodule contours (up to 4 readers) "
           "plus 1-5 malignancy ratings. A binary nodule mask is the common "
           "target; malignancy can be a separate regression/classification "
           "head. Consensus masks are typically built at a >=50% agreement "
           "level (e.g. via pylidc)."),
)

# ---------------------------------------------------------------------------
# CT / Head — RSNA Intracranial Hemorrhage (slice labels only -> Grad-CAM)
# ---------------------------------------------------------------------------
CT_HEAD_ICH = ModelSpec(
    key="ct_head_ich_classifier",
    dataset="RSNA Intracranial Hemorrhage Detection",
    modality="CT", body_part="HEAD",
    task=TaskType.CLASSIFICATION,
    localization=Localization.GRADCAM,
    multilabel=True,   # subtypes are not mutually exclusive; 'any' is a union
    class_map={
        0: "any",
        1: "epidural",
        2: "intraparenchymal",
        3: "intraventricular",
        4: "subarachnoid",
        5: "subdural",
    },
    recommended_arch="2.5D / 3D CNN classifier (e.g. ResNet/EfficientNet on "
                     "windowed slices) with multi-label sigmoid head",
    notes=("Base dataset is SLICE-LEVEL classification only — no masks or "
           "boxes. Localization must be weakly-supervised (Grad-CAM). Use "
           "brain/subdural/bone window stacks as 3-channel input. If you need "
           "true segmentation, add a mask dataset (e.g. BHSD / PhysioNet ICH)."),
)

# ---------------------------------------------------------------------------
# MRI / Brain — BraTS (voxel masks -> supervised localization)
# ---------------------------------------------------------------------------
MR_BRAIN_TUMOR = ModelSpec(
    key="mr_brain_tumor_seg",
    dataset="BraTS (Brain Tumor Segmentation)",
    modality="MR", body_part="BRAIN",
    task=TaskType.SEGMENTATION,
    localization=Localization.MASK_CC,
    # These are the three NESTED evaluation regions the model outputs. The raw
    # voxel integer labels differ (see notes) — do not confuse the two.
    class_map={
        0: "background",
        1: "whole_tumor",       # WT = union of all tumor tissue
        2: "tumor_core",        # TC = enhancing + necrotic core
        3: "enhancing_tumor",   # ET
    },
    multilabel=True,   # nested regions overlap -> sigmoid per region, not softmax
    recommended_arch="UNETR / SwinUNETR / 3D U-Net (monai.networks.nets)",
    modalities_required=["T1", "T1Gd(T1ce)", "T2", "FLAIR"],
    notes=("RAW on-disk voxel labels (BraTS <=2021 / MSD differ!): "
           "{1: NCR/NET necrotic core, 2: ED edema, 4: ET enhancing}, label 3 "
           "unused. BraTS 2023+ RELABELED enhancing tumor from 4 -> 3. "
           "Models are scored on nested regions WT(1+2+4), TC(1+4), ET(4). "
           "Confirm the integer scheme of YOUR downloaded release before "
           "building the target tensor."),
)

# ---------------------------------------------------------------------------
# Ultrasound / Breast — BUSI (masks available -> supervised localization)
# ---------------------------------------------------------------------------
US_BREAST_LESION = ModelSpec(
    key="us_breast_lesion",
    dataset="BUSI (Breast Ultrasound Images)",
    modality="US", body_part="BREAST",
    task=TaskType.DETECTION,
    localization=Localization.MASK_CC,
    class_map={0: "background", 1: "benign", 2: "malignant"},
    recommended_arch="Mask R-CNN (torchvision) or U-Net segmentation head",
    notes=("BUSI: 780 grayscale images in 3 categories (normal / benign / "
           "malignant) WITH ground-truth lesion masks, so localization is "
           "supervised. 'normal' images contain no lesion (empty mask)."),
)

# ---------------------------------------------------------------------------
# X-Ray / Chest — CheXpert / MIMIC-CXR (image labels only -> Grad-CAM)
# ---------------------------------------------------------------------------
CXR_MULTILABEL = ModelSpec(
    key="cxr_multilabel_classifier",
    dataset="CheXpert / MIMIC-CXR",
    modality="XR", body_part="CHEST",
    task=TaskType.CLASSIFICATION,
    localization=Localization.GRADCAM,
    multilabel=True,
    # The canonical CheXpert 14 observations (MIMIC-CXR shares this schema).
    class_map={
        0: "No Finding",
        1: "Enlarged Cardiomediastinum",
        2: "Cardiomegaly",
        3: "Lung Opacity",
        4: "Lung Lesion",
        5: "Edema",
        6: "Consolidation",
        7: "Pneumonia",
        8: "Atelectasis",
        9: "Pneumothorax",
        10: "Pleural Effusion",
        11: "Pleural Other",
        12: "Fracture",
        13: "Support Devices",
    },
    recommended_arch="RAD-DINO (microsoft/rad-dino) frozen encoder + linear "
                     "multi-label head; or DenseNet-121 (CheXNet-style)",
    notes=("Image-level multi-label only — no boxes in base labels. CheXpert "
           "uses uncertainty labels {0,1,-1(uncertain),blank}; choose a policy "
           "(U-Ones / U-Zeros / ignore) per class when building targets. "
           "Localization is weakly-supervised via Grad-CAM."),
)


# Registry keyed by the string returned from dicom_ingestion.route_to_model().
MODEL_SPECS: Dict[str, ModelSpec] = {
    spec.key: spec for spec in (
        CT_CHEST_NODULE,
        CT_HEAD_ICH,
        MR_BRAIN_TUMOR,
        US_BREAST_LESION,
        CXR_MULTILABEL,
    )
}


def get_spec(key: str) -> Optional[ModelSpec]:
    """Look up a ModelSpec by registry key, or None if unmapped."""
    return MODEL_SPECS.get(key)


# ---------------------------------------------------------------------------
# UI-facing categories. Letting a user pick the category FORCES routing to the
# right specialized model regardless of (often missing/wrong) DICOM metadata —
# the single most reliable way to get modality-specific anomaly detection.
# ---------------------------------------------------------------------------
CATEGORY_TO_KEY: Dict[str, str] = {
    "ct_chest":  "ct_chest_nodule_seg",
    "ct_head":   "ct_head_ich_classifier",
    "mr_brain":  "mr_brain_tumor_seg",
    "us_breast": "us_breast_lesion",
    "cxr":       "cxr_multilabel_classifier",
}


def route_by_category(category: str) -> str:
    """
    Map a UI category id to its registry key. Raises ValueError on unknown
    category so the caller can surface a clear error instead of silently
    falling back to a generic model.
    """
    try:
        return CATEGORY_TO_KEY[category]
    except KeyError as exc:
        raise ValueError(
            f"Unknown category '{category}'. Valid: {sorted(CATEGORY_TO_KEY)}"
        ) from exc


if __name__ == "__main__":  # pragma: no cover
    for k, s in MODEL_SPECS.items():
        loc = "supervised mask" if s.localization is Localization.MASK_CC \
            else "Grad-CAM (weak)"
        print(f"{k:28s} | {s.dataset:35s} | {s.task.value:14s} | "
              f"{s.num_classes} cls | loc={loc}")
