"""
explainability.py
==================================================================
Weakly-supervised localization + explainability for the CLASSIFICATION models
(RSNA-ICH, CheXpert/MIMIC-CXR) whose datasets provide no boxes or masks.

Two pieces:
  1. ``heatmap_to_boxes`` — pure post-processing: turn any 2D/3D saliency map
     into bounding boxes via threshold + connected components. Reuses the exact
     bbox logic from ConfidenceLocalizer so segmentation and Grad-CAM paths
     yield identically-shaped ``Detection`` outputs.
  2. ``GradCAMLocalizer`` — wraps pytorch-grad-cam to compute the saliency map
     for a chosen target class against a chosen model layer, then boxes it.

For SEGMENTATION models (LIDC/BraTS/BUSI) you do NOT need this — use
ConfidenceLocalizer directly on the mask logits. This module exists purely so
the image-level classifiers can still answer "where to look".
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from confidence_localization import ConfidenceLocalizer, Detection

logger = logging.getLogger("explainability")
logger.addHandler(logging.NullHandler())

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# 1. Saliency map -> bounding boxes (pure, testable, no model needed)
# ---------------------------------------------------------------------------
def heatmap_to_boxes(
    heatmap: ArrayLike,
    class_id: int,
    rel_threshold: float = 0.5,
    min_region_voxels: int = 20,
) -> List[Detection]:
    """
    Convert a Grad-CAM saliency map into ``Detection`` boxes.

    The heatmap is min-max normalised to [0, 1] first, so ``rel_threshold`` is
    a *relative* activation level (0.5 = half of the map's own peak), which is
    the standard way weak-localization boxes are drawn from CAMs.

    Parameters
    ----------
    heatmap : 2D (H, W) or 3D (D, H, W) non-negative saliency map.
    class_id : the class this heatmap explains (recorded on each Detection).
    rel_threshold : fraction of the normalised peak used to binarise.
    min_region_voxels : drop components smaller than this.

    Returns
    -------
    List[Detection] sorted most-salient first. ``confidence`` / ``peak_confidence``
    here are *saliency* magnitudes (0-1), NOT calibrated probabilities — the
    class probability comes from the classifier head separately.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim not in (2, 3):
        raise ValueError(f"heatmap must be 2D or 3D, got shape {heatmap.shape}")

    # Robust min-max normalise to [0, 1]; guard against a flat map.
    hmin, hmax = float(heatmap.min()), float(heatmap.max())
    if hmax - hmin < 1e-8:
        logger.debug("Flat heatmap for class %d — no boxes.", class_id)
        return []
    norm = (heatmap - hmin) / (hmax - hmin)

    # Reuse ConfidenceLocalizer's connected-component + bbox machinery by
    # treating the normalised saliency as a single-class "probability" map.
    localizer = ConfidenceLocalizer(
        mode="sigmoid",
        prob_threshold=float(rel_threshold),
        min_region_voxels=min_region_voxels,
    )
    dets = localizer._localise_class(class_id, norm)  # noqa: SLF001 (intended reuse)
    dets.sort(key=lambda d: d.peak_confidence, reverse=True)
    return dets


# ---------------------------------------------------------------------------
# 1b. Dependency-free Grad-CAM (works for 2D and 3D conv nets)
# ---------------------------------------------------------------------------
def make_vit_reshape_transform(num_prefix_tokens: int = 1):
    """
    Build a reshape_transform for ViT backbones: turn a ``(B, tokens, dim)``
    hidden state into a ``(B, dim, h, w)`` spatial grid by dropping the prefix
    tokens (CLS + any register tokens) and folding the square patch grid.

    For ``microsoft/rad-dino`` (plain DINOv2, no registers) use
    ``num_prefix_tokens=1``; for DINOv2-with-registers use
    ``1 + config.num_register_tokens``.
    """
    def _reshape(t: "torch.Tensor") -> "torch.Tensor":
        t = t[:, num_prefix_tokens:, :]                 # drop CLS(+registers)
        n = t.shape[1]
        h = w = int(round(n ** 0.5))
        if h * w != n:
            raise ValueError(f"{n} patch tokens is not a square grid")
        return t.reshape(t.shape[0], h, w, t.shape[2]).permute(0, 3, 1, 2)
    return _reshape


def compute_gradcam(
    model: "torch.nn.Module",
    input_tensor: "torch.Tensor",
    target_layer: "torch.nn.Module",
    target_class: int,
    multilabel: bool = False,
    reshape_transform=None,
) -> np.ndarray:
    """
    Minimal Grad-CAM (Selvaraju et al., 2017) with no external dependency.

    Hooks ``target_layer`` (last conv block for CNNs, or the last transformer
    block's ``norm1`` for ViTs) to capture activations and their gradients
    w.r.t. the target class score, then forms the ReLU-weighted class
    activation map.

    For conv nets leave ``reshape_transform=None`` (activations are already
    ``(B, C, *spatial)``). For ViT backbones pass
    :func:`make_vit_reshape_transform` so the ``(B, tokens, dim)`` hidden state
    is folded to ``(B, dim, h, w)`` before weighting.

    Returns a saliency map (spatial shape of the feature grid), normalised
    downstream by :func:`heatmap_to_boxes`.
    """
    model.eval()
    # A frozen backbone (all params requires_grad=False) means no gradient
    # reaches the hooked activation unless the *input* carries grad. Enabling it
    # is universally safe and required for Grad-CAM over frozen encoders.
    input_tensor = input_tensor.detach().clone().requires_grad_(True)
    activations: dict = {}
    gradients: dict = {}

    def fwd_hook(_m, _inp, out):
        activations["value"] = out if isinstance(out, torch.Tensor) else out[0]

    def bwd_hook(_m, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(input_tensor)
        score = (logits.sigmoid() if multilabel else logits.softmax(dim=1))[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward()

        acts = activations["value"]        # (B, ...)
        grads = gradients["value"]         # (B, ...)
        if reshape_transform is not None:
            acts = reshape_transform(acts)     # (B, C, h, w)
            grads = reshape_transform(grads)
        acts, grads = acts[0], grads[0]        # drop batch -> (C, *spatial)
    finally:
        h1.remove()
        h2.remove()

    # Channel weights = global-average-pooled gradients over spatial dims.
    spatial_dims = tuple(range(1, acts.ndim))
    weights = grads.mean(dim=spatial_dims, keepdim=True)   # (C, 1, 1[, 1])
    cam = torch.relu((weights * acts).sum(dim=0))          # (*spatial)
    return cam.detach().cpu().numpy()



class GradCAMLocalizer:
    """
    Compute Grad-CAM for a classification model and box the salient regions.

    Parameters
    ----------
    model : a trained classification model in eval mode.
    target_layers : list of layers to hook (typically the last conv block, or
        for RAD-DINO/ViT the last attention block's norm layer with a ViT
        reshape transform — see ``reshape_transform``).
    reshape_transform : optional callable for transformer backbones that turns
        token sequences back into a spatial grid (required for ViT/RAD-DINO).
    use_cuda : run on GPU if available.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        target_layers: Sequence["torch.nn.Module"],
        reshape_transform=None,
        use_cuda: Optional[bool] = None,
    ) -> None:
        try:
            from pytorch_grad_cam import GradCAM  # lazy: only needed here
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pytorch-grad-cam is required for GradCAMLocalizer. "
                "Install it: pip install grad-cam"
            ) from exc

        self.model = model.eval()
        device = ("cuda" if (use_cuda if use_cuda is not None
                             else torch.cuda.is_available()) else "cpu")
        self.device = device
        self.model.to(device)
        self._cam = GradCAM(
            model=self.model,
            target_layers=list(target_layers),
            reshape_transform=reshape_transform,
        )

    def localize(
        self,
        input_tensor: "torch.Tensor",
        target_class: int,
        rel_threshold: float = 0.5,
        min_region_voxels: int = 20,
    ) -> List[Detection]:
        """
        Run Grad-CAM for ``target_class`` and return boxed salient regions.

        input_tensor : (1, C, H, W) preprocessed batch of one image.
        """
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        input_tensor = input_tensor.to(self.device)
        # grayscale_cam: (batch, H, W) in [0, 1]; take the single sample.
        try:
            grayscale_cam = self._cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(target_class)],
            )[0]
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Grad-CAM computation failed: {exc}") from exc

        return heatmap_to_boxes(
            grayscale_cam,
            class_id=target_class,
            rel_threshold=rel_threshold,
            min_region_voxels=min_region_voxels,
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    # Synthetic 2D saliency map with two hot spots — no model/package needed.
    rng = np.random.default_rng(1)
    hm = rng.random((128, 128)).astype(np.float32) * 0.2
    hm[20:40, 30:55] += 0.9      # strong focus 1
    hm[80:95, 90:110] += 0.7     # focus 2

    boxes = heatmap_to_boxes(hm, class_id=7, rel_threshold=0.5,
                             min_region_voxels=30)
    print(f"Derived {len(boxes)} box(es) from Grad-CAM heatmap:")
    for i, d in enumerate(boxes, 1):
        print(f"  [{i}] class={d.class_id} peak_saliency={d.peak_confidence:.3f} "
              f"pixels={d.voxel_count} bbox(y0,y1,x0,x1)={d.bbox}")
