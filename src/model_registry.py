"""
model_registry.py
==================================================================
Maps a routed registry key -> a concrete torch model, built from its
:class:`ModelSpec` (correct output-channel count, input channels, and
spatial dimensionality). For the smoke test the weights are RANDOMLY
INITIALIZED — outputs are meaningless, the point is to prove the
ingest -> route -> infer -> localize plumbing runs with correct shapes.

To load real weights later: implement ``load_weights(key, path)`` to
``model.load_state_dict(...)`` after ``build``. Architectures here match the
recommendations in ARCHITECTURE.md.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from label_taxonomies import ModelSpec, MODEL_SPECS, TaskType, Localization, get_spec

logger = logging.getLogger("model_registry")
logger.addHandler(logging.NullHandler())

# Segmentation backbone from MONAI if available; otherwise a tiny fallback so
# the smoke test still runs shape-correctly without MONAI installed.
try:
    from monai.networks.nets import UNet as _MonaiUNet
    _HAVE_MONAI = True
except Exception:  # pragma: no cover
    _HAVE_MONAI = False


# ---------------------------------------------------------------------------
# Helpers to derive geometry from a spec
# ---------------------------------------------------------------------------
def _spatial_dims(spec: ModelSpec) -> int:
    """CT/MR are volumetric (3D); US/XR are projection images (2D)."""
    return 3 if spec.modality in ("CT", "MR") else 2


def _in_channels(spec: ModelSpec) -> int:
    """BraTS stacks 4 MR modalities as channels; everything else is grayscale."""
    return max(1, len(spec.modalities_required)) if spec.modalities_required else 1


# ---------------------------------------------------------------------------
# Classification stand-in (real conv layer exposed for Grad-CAM)
# ---------------------------------------------------------------------------
class SmokeCNNClassifier(nn.Module):
    """
    Small conv classifier for the smoke test. Exposes ``feature_layer`` (the
    last conv) so Grad-CAM can hook a spatially-resolved activation map. Works
    in 2D or 3D depending on ``spatial_dims``.
    """

    def __init__(self, spatial_dims: int, in_channels: int, num_classes: int,
                 widths=(8, 16, 32)) -> None:
        super().__init__()
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        Pool = nn.MaxPool3d if spatial_dims == 3 else nn.MaxPool2d
        gap = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)

        blocks = []
        c = in_channels
        last_conv: Optional[nn.Module] = None
        for w in widths:
            conv = Conv(c, w, kernel_size=3, padding=1)
            last_conv = conv
            blocks += [conv, nn.ReLU(), Pool(2)]
            c = w
        self.features = nn.Sequential(*blocks)
        self.feature_layer = last_conv        # Grad-CAM target
        self.gap = gap
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        pooled = self.gap(f).flatten(1)
        return self.head(pooled)


class _TinySegNet(nn.Module):
    """Fallback segmentation net (only used if MONAI is unavailable)."""

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.net = nn.Sequential(
            Conv(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True),
            Conv(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            Conv(16, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_segmentation(spec: ModelSpec) -> nn.Module:
    sd, ic, oc = _spatial_dims(spec), _in_channels(spec), spec.num_classes
    if _HAVE_MONAI:
        # Standard small U-Net; depth/strides fine for the smoke-test volumes.
        return _MonaiUNet(
            spatial_dims=sd, in_channels=ic, out_channels=oc,
            channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
        )
    logger.warning("MONAI unavailable — using _TinySegNet fallback for %s", spec.key)
    return _TinySegNet(sd, ic, oc)


def _build_classifier(spec: ModelSpec) -> nn.Module:
    return SmokeCNNClassifier(_spatial_dims(spec), _in_channels(spec), spec.num_classes)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    """
    Lazily builds and caches one model per registry key.

    Parameters
    ----------
    device : "cpu"/"cuda"; defaults to CUDA when available.
    seed : RNG seed for reproducible random initialisation.
    """

    def __init__(self, device: Optional[str] = None, seed: int = 0) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self._cache: Dict[str, nn.Module] = {}

    def build(self, key: str) -> nn.Module:
        """Build (or return cached) model for ``key``. Raises on unknown key."""
        if key in self._cache:
            return self._cache[key]
        spec = get_spec(key)
        if spec is None:
            raise KeyError(
                f"No ModelSpec registered for key '{key}'. "
                f"Known keys: {sorted(MODEL_SPECS)}"
            )
        torch.manual_seed(self.seed)  # reproducible random init

        if spec.localization == Localization.MASK_CC:
            model = _build_segmentation(spec)   # seg/detection -> mask logits
        else:
            model = _build_classifier(spec)     # classification -> class logits

        model = model.to(self.device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Built '%s' (%s, %s) — %.2fM params",
                    key, spec.recommended_arch.split("(")[0].strip(),
                    type(model).__name__, n_params / 1e6)
        self._cache[key] = model
        return model

    def load_weights(self, key: str, checkpoint_path: str) -> nn.Module:
        """Load real weights into a built model (for production use)."""
        model = self.build(key)
        state = torch.load(checkpoint_path, map_location=self.device)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        logger.info("Loaded weights for '%s' from %s", key, checkpoint_path)
        return model


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")
    reg = ModelRegistry(device="cpu")
    for k in MODEL_SPECS:
        m = reg.build(k)
        spec = get_spec(k)
        print(f"{k:28s} -> {type(m).__name__:18s} "
              f"in={_in_channels(spec)} out={spec.num_classes} dims={_spatial_dims(spec)}D")
