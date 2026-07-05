"""
confidence_localization.py
==================================================================
Turn raw model logits into (a) a calibrated confidence score and
(b) spatial localisation (bounding boxes) of high-probability regions.

Works for both:
  * 2D segmentation logits   -> shape (C, H, W)
  * 3D segmentation logits   -> shape (C, D, H, W)

Two probability modes
----------------------
  * "softmax"  : mutually-exclusive classes (channel 0 = background).
  * "sigmoid"  : independent / multi-label channels.

Calibration
-----------
Raw softmax/sigmoid outputs are typically *over-confident*. We apply optional
**temperature scaling** (Guo et al., 2017): divide logits by a scalar T>1
learned on a validation set. T=1.0 is a no-op, so the class is safe to use
before you have fit a temperature.

Everything here is pure tensor math + connected-component analysis. It makes
no clinical claim — it reports where the model's probability mass is and how
peaked it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

logger = logging.getLogger("confidence_localization")
logger.addHandler(logging.NullHandler())

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single localised region for one class."""
    class_id: int
    # bbox is axis-min/axis-max per spatial dim, i.e. for 3D:
    #   (z_min, z_max, y_min, y_max, x_min, x_max); for 2D the z pair is absent.
    bbox: Tuple[int, ...]
    confidence: float          # calibrated mean probability inside the region
    peak_confidence: float     # calibrated max probability inside the region
    voxel_count: int           # region size (voxels/pixels)

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InferenceResult:
    """Full result for one scan: per-class detections + overall confidence."""
    detections: List[Detection]
    # Global per-class confidence = max calibrated prob anywhere for that class.
    class_confidence: Dict[int, float]
    top_class: Optional[int]
    top_confidence: float


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class ConfidenceLocalizer:
    """
    Convert segmentation logits -> calibrated confidence + bounding boxes.

    Parameters
    ----------
    mode : "softmax" (mutually exclusive) or "sigmoid" (multi-label).
    prob_threshold : probability above which a voxel is "positive".
    temperature : temperature-scaling scalar T (>0). Fit on validation data;
        default 1.0 leaves logits unchanged.
    min_region_voxels : discard connected components smaller than this
        (removes speckle / single-voxel false positives).
    background_index : channel treated as background in softmax mode (ignored
        for sigmoid mode).
    """

    def __init__(
        self,
        mode: str = "softmax",
        prob_threshold: float = 0.5,
        temperature: float = 1.0,
        min_region_voxels: int = 10,
        background_index: int = 0,
    ) -> None:
        if mode not in ("softmax", "sigmoid"):
            raise ValueError(f"mode must be 'softmax' or 'sigmoid', got {mode!r}")
        if not (0.0 < prob_threshold < 1.0):
            raise ValueError("prob_threshold must be in (0, 1)")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.mode = mode
        self.prob_threshold = float(prob_threshold)
        self.temperature = float(temperature)
        self.min_region_voxels = int(min_region_voxels)
        self.background_index = int(background_index)

    # -- probability conversion --------------------------------------------
    def _to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling then softmax/sigmoid.

        Expects logits shaped (C, *spatial). Returns probabilities of the
        same shape.
        """
        if logits.ndim < 2:
            raise ValueError(
                f"Expected logits of shape (C, *spatial), got {tuple(logits.shape)}"
            )
        scaled = logits / self.temperature
        if self.mode == "softmax":
            # Softmax over the channel axis (dim=0).
            return F.softmax(scaled, dim=0)
        return torch.sigmoid(scaled)

    # -- bounding boxes -----------------------------------------------------
    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, ...]:
        """
        Return (min, max) index pairs per axis for a boolean mask.
        For an N-D mask returns a flat tuple of length 2*N ordered
        (axis0_min, axis0_max, axis1_min, axis1_max, ...). ``max`` is
        inclusive (the last positive index).
        """
        coords = np.array(np.nonzero(mask))  # shape (ndim, n_positive)
        bounds: List[int] = []
        for axis in range(mask.ndim):
            axis_coords = coords[axis]
            bounds.extend((int(axis_coords.min()), int(axis_coords.max())))
        return tuple(bounds)

    def _localise_class(
        self, class_id: int, prob_map: np.ndarray
    ) -> List[Detection]:
        """
        For one class probability map, threshold -> label connected components
        -> emit a Detection per surviving component.
        """
        binary = prob_map >= self.prob_threshold
        if not binary.any():
            return []

        # Label 6/26-connected (2D/3D) components. Default structure = full
        # connectivity for the array's dimensionality.
        labeled, n_components = ndimage.label(binary)
        detections: List[Detection] = []
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            voxel_count = int(comp_mask.sum())
            if voxel_count < self.min_region_voxels:
                continue
            region_probs = prob_map[comp_mask]
            detections.append(
                Detection(
                    class_id=class_id,
                    bbox=self._bbox_from_mask(comp_mask),
                    confidence=float(region_probs.mean()),
                    peak_confidence=float(region_probs.max()),
                    voxel_count=voxel_count,
                )
            )
        return detections

    # -- public API ---------------------------------------------------------
    def __call__(self, logits: ArrayLike) -> InferenceResult:
        """
        Run the full pipeline on a single sample.

        ``logits`` may be numpy or torch, shaped (C, *spatial) or
        (1, C, *spatial) (a leading batch of size 1 is squeezed).
        """
        # ---- normalise input to a (C, *spatial) torch.Tensor -------------
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a numpy array or torch tensor")
        logits = logits.detach().float()

        if logits.ndim >= 2 and logits.shape[0] == 1 and logits.ndim > 3:
            # Squeeze a leading batch dim of 1 if present, e.g. (1,C,D,H,W).
            logits = logits.squeeze(0)

        try:
            probs = self._to_probabilities(logits)
        except Exception as exc:
            raise RuntimeError(f"Probability conversion failed: {exc}") from exc

        probs_np = probs.cpu().numpy()
        n_classes = probs_np.shape[0]

        detections: List[Detection] = []
        class_confidence: Dict[int, float] = {}

        for class_id in range(n_classes):
            if self.mode == "softmax" and class_id == self.background_index:
                continue  # never localise/score the background channel
            prob_map = probs_np[class_id]
            # Global confidence for this class = its peak probability anywhere.
            class_confidence[class_id] = float(prob_map.max())
            detections.extend(self._localise_class(class_id, prob_map))

        if class_confidence:
            top_class = max(class_confidence, key=class_confidence.get)
            top_confidence = class_confidence[top_class]
        else:
            top_class, top_confidence = None, 0.0

        # Sort detections most-confident first for convenient consumption.
        detections.sort(key=lambda d: d.peak_confidence, reverse=True)
        return InferenceResult(
            detections=detections,
            class_confidence=class_confidence,
            top_class=top_class,
            top_confidence=top_confidence,
        )

    # -- calibration note ---------------------------------------------------
    # Temperature is APPLIED here (see _to_probabilities). FITTING a temperature
    # and measuring calibration (ECE) live in calibration.py:
    #     from calibration import fit_temperature, expected_calibration_error
    #     T = fit_temperature(val_logits, val_labels, mode="multilabel")
    #     loc = ConfidenceLocalizer(mode="sigmoid", temperature=T)


# ---------------------------------------------------------------------------
# Demo with a MOCK model output (no real weights involved)
# ---------------------------------------------------------------------------
def _make_mock_logits() -> torch.Tensor:
    """
    Build synthetic 3D segmentation logits (C=3, D=16, H=64, W=64) with two
    planted 'lesion' blobs so the localiser has something to find. Purely for
    exercising the code path — not a model and not medical data.
    """
    rng = np.random.default_rng(0)
    c, d, h, w = 3, 16, 64, 64
    logits = rng.normal(0.0, 1.0, size=(c, d, h, w)).astype(np.float32)
    # Bias background channel up everywhere, then carve out two positive blobs.
    logits[0] += 2.0
    logits[1, 4:9, 10:22, 12:26] += 6.0     # class-1 blob
    logits[2, 8:13, 40:52, 38:50] += 5.0    # class-2 blob
    return torch.from_numpy(logits)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    localizer = ConfidenceLocalizer(
        mode="softmax", prob_threshold=0.5, temperature=1.0, min_region_voxels=25
    )
    result = localizer(_make_mock_logits())

    print(f"Top class      : {result.top_class} "
          f"(confidence {result.top_confidence:.3f})")
    print(f"Per-class conf : "
          + ", ".join(f"{k}:{v:.3f}" for k, v in result.class_confidence.items()))
    print(f"Detections     : {len(result.detections)}")
    for i, det in enumerate(result.detections, 1):
        print(f"  [{i}] class={det.class_id} conf={det.confidence:.3f} "
              f"peak={det.peak_confidence:.3f} voxels={det.voxel_count} "
              f"bbox(z0,z1,y0,y1,x0,x1)={det.bbox}")
