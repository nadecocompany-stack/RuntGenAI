"""
cxr_rad_dino.py
==================================================================
Wires the ``microsoft/rad-dino`` chest-X-ray encoder into the platform's
classification + weak-localization path (registry key
``cxr_multilabel_classifier``).

RAD-DINO facts (from the model card) this module relies on:
  * Loaded as a plain DINOv2 encoder: ``AutoModel.from_pretrained("microsoft/rad-dino")``.
  * Input 518x518, patch 14 -> 37x37 = 1369 patch tokens (+1 CLS, NO register
    tokens). ``last_hidden_state`` is ``(B, 1370, 768)``; token 0 is CLS.
  * Downstream classification = a head on the CLS token; dense/localization
    uses the patch tokens reshaped to ``(B, 768, 37, 37)``.

Grad-CAM on a ViT: we hook the last block's ``norm1`` and fold the token
sequence back to the patch grid via ``make_vit_reshape_transform`` (dropping
the single CLS prefix token). Verified to yield non-degenerate maps.

OFFLINE NOTE: ``from_pretrained`` needs network access to the HuggingFace hub.
Where that's unavailable (e.g. CI sandboxes), ``build_rad_dino_cxr(pretrained=
False)`` constructs the *same* DINOv2 architecture with random weights so the
wiring can be exercised; swap to ``pretrained=True`` for real inference.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from explainability import (
    compute_gradcam,
    heatmap_to_boxes,
    make_vit_reshape_transform,
    Detection,
)

logger = logging.getLogger("cxr_rad_dino")
logger.addHandler(logging.NullHandler())

RAD_DINO_HF_NAME = "microsoft/rad-dino"
RAD_DINO_HIDDEN = 768
RAD_DINO_PATCH = 14
RAD_DINO_IMAGE = 518


class RadDinoCXRClassifier(nn.Module):
    """
    RAD-DINO encoder + a multi-label classification head on the CLS token.

    Parameters
    ----------
    num_classes : number of findings (14 for the CheXpert/MIMIC schema).
    freeze_backbone : if True (recommended — the model card notes fine-tuning is
        usually unnecessary), only the head is trained.
    pretrained : load real RAD-DINO weights from the hub. Set False for an
        offline random-weight build with identical architecture.
    image_size : only used when ``pretrained=False`` to size the random config
        (smaller = faster tests). Real weights always use 518.
    """

    def __init__(
        self,
        num_classes: int = 14,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        image_size: int = RAD_DINO_IMAGE,
    ) -> None:
        super().__init__()
        self.backbone = self._load_backbone(pretrained, image_size)
        hidden = self.backbone.config.hidden_size
        # Prefix tokens to skip when folding to a grid: CLS + any registers.
        self.num_prefix_tokens = 1 + int(
            getattr(self.backbone.config, "num_register_tokens", 0)
        )
        self.head = nn.Linear(hidden, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def _load_backbone(pretrained: bool, image_size: int) -> nn.Module:
        try:
            from transformers import AutoModel, Dinov2Model, Dinov2Config
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for RAD-DINO. pip install transformers"
            ) from exc

        if pretrained:
            logger.info("Loading pretrained RAD-DINO from %s", RAD_DINO_HF_NAME)
            return AutoModel.from_pretrained(RAD_DINO_HF_NAME)

        logger.warning(
            "Building RAD-DINO architecture with RANDOM weights (offline). "
            "Outputs are meaningless until real weights are loaded."
        )
        cfg = Dinov2Config(
            hidden_size=RAD_DINO_HIDDEN,
            num_hidden_layers=12,
            num_attention_heads=12,
            patch_size=RAD_DINO_PATCH,
            image_size=image_size,
            mlp_ratio=4,
        )
        return Dinov2Model(cfg)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, H, W) -> logits (B, num_classes)."""
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0]     # RAD-DINO: token 0 = CLS
        return self.head(cls_token)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Last transformer block's norm1 — the robust ViT Grad-CAM target."""
        return self.backbone.encoder.layer[-1].norm1


def build_rad_dino_cxr(
    num_classes: int = 14,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    image_size: int = RAD_DINO_IMAGE,
    device: Optional[str] = None,
) -> RadDinoCXRClassifier:
    """Construct + move a RAD-DINO CXR classifier to ``device`` in eval mode."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RadDinoCXRClassifier(
        num_classes=num_classes, freeze_backbone=freeze_backbone,
        pretrained=pretrained, image_size=image_size,
    ).to(device).eval()
    return model


def preprocess_cxr(images, device: Optional[str] = None) -> torch.Tensor:
    """
    Preprocess a chest X-ray (PIL image or list) with RAD-DINO's own image
    processor (resize/crop to 518 + the model's normalisation). Requires the
    hub-downloaded processor; for offline tests feed a tensor directly instead.
    """
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(RAD_DINO_HF_NAME)
    batch = processor(images=images, return_tensors="pt")
    pv = batch["pixel_values"]
    return pv.to(device) if device else pv


def localize_cxr(
    model: RadDinoCXRClassifier,
    pixel_values: torch.Tensor,
    target_class: int,
    rel_threshold: float = 0.5,
    min_region_pixels: int = 4,
    to_pixel_coords: bool = True,
) -> List[Detection]:
    """
    Grad-CAM localization for one finding: fold the ViT tokens to the patch
    grid, build the saliency map, and derive bounding boxes.

    With ``to_pixel_coords=True`` (default) boxes are rescaled from the 37x37
    patch grid to the model's input-pixel space (e.g. 518x518) via the coordinate
    mapper, so coordinates are usable on the processed image. Map further back to
    the original CXR by the AutoImageProcessor's resize ratio if needed.
    """
    reshape = make_vit_reshape_transform(model.num_prefix_tokens)
    cam = compute_gradcam(
        model, pixel_values, model.gradcam_target_layer, target_class,
        multilabel=True, reshape_transform=reshape,
    )
    boxes = heatmap_to_boxes(cam, class_id=target_class,
                             rel_threshold=rel_threshold,
                             min_region_voxels=min_region_pixels)
    if not to_pixel_coords:
        return boxes

    from coordinate_mapping import map_detection_bbox
    input_hw = tuple(int(s) for s in pixel_values.shape[2:])   # (H, W)
    mapped = []
    for b in boxes:
        px = map_detection_bbox(b.bbox, feature_shape=cam.shape,
                                input_shape=input_hw)["input_voxel"]
        mapped.append(Detection(class_id=b.class_id, bbox=tuple(px),
                                confidence=b.confidence,
                                peak_confidence=b.peak_confidence,
                                voxel_count=b.voxel_count))
    return mapped


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")
    # Offline demo: random-weight RAD-DINO at reduced size for a fast CPU run.
    model = build_rad_dino_cxr(num_classes=14, pretrained=False,
                               image_size=224, device="cpu")
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)
    probs = logits.sigmoid()[0].detach()

    # Pick the finding whose Grad-CAM is most peaked (skip index 0 = No Finding).
    from explainability import compute_gradcam, make_vit_reshape_transform
    reshape = make_vit_reshape_transform(model.num_prefix_tokens)
    peaks = [(c, float(compute_gradcam(model, x, model.gradcam_target_layer, c,
                                       multilabel=True, reshape_transform=reshape).max()))
             for c in range(1, 14)]
    target = max(peaks, key=lambda t: t[1])[0]

    boxes = localize_cxr(model, x, target, min_region_pixels=2)
    print(f"logits {tuple(logits.shape)} | strongest finding class {target} "
          f"p={float(probs[target]):.2f} | boxes={len(boxes)}")
    for b in boxes:
        print(f"  class={b.class_id} peak={b.peak_confidence:.2f} "
              f"patches={b.voxel_count} bbox(y0,y1,x0,x1)={b.bbox}")
