"""
inference.py
==================================================================
Memory-bounded volumetric inference. Real CT/MR volumes (e.g. 512x512x300+)
do not fit a full forward pass on most GPUs. ``sliding_window_inference`` runs
the network on fixed-size overlapping ROIs and blends the results, so peak
memory scales with the *patch*, not the whole volume.

Two side benefits over a naive whole-volume ``model(x)``:
  * No ``DivisiblePad`` hack — each ROI is a fixed size chosen divisible by the
    network's stride product, so U-Net skip-connections always line up.
  * The output is full-resolution and identical in spatial size to the input,
    which means segmentation boxes are already in preprocessed-voxel space
    (no pad offset to undo before coordinate mapping).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

from monai.inferers import sliding_window_inference


def default_roi_size(
    spatial_shape: Sequence[int], base: int = 96, k: int = 16
) -> Tuple[int, ...]:
    """
    Pick a per-axis ROI: the smaller of ``base`` and the volume dimension
    rounded up to a multiple of ``k`` (the stride product), with a floor of
    ``k`` so the network always has enough spatial extent to downsample.

    e.g. (45,45,8) -> (48,48,16);  (512,512,300) -> (96,96,96).
    """
    return tuple(
        int(min(base, max(k, math.ceil(d / k) * k))) for d in spatial_shape
    )


def segment_volume(
    model: "torch.nn.Module",
    x: "torch.Tensor",
    *,
    roi_size: Optional[Sequence[int]] = None,
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    mode: str = "gaussian",
    use_amp: bool = False,
    device: Optional[str] = None,
) -> "torch.Tensor":
    """
    Run a segmentation model over a volume with sliding-window inference.

    Parameters
    ----------
    x : (B, C, *spatial) input.
    roi_size : patch size (defaults to :func:`default_roi_size`). Must be
        divisible by the network's stride product.
    sw_batch_size : ROIs per forward pass — raise to trade memory for speed.
    overlap : fractional ROI overlap (0.25 is a good default; higher = smoother
        seams, slower).
    mode : blending window, "gaussian" (weights centre voxels more) or
        "constant".
    use_amp : autocast (fp16/bf16) to roughly halve activation memory.

    Returns
    -------
    logits (B, num_classes, *spatial) at full input resolution.
    """
    if x.ndim < 4:
        raise ValueError(f"expected (B, C, *spatial), got shape {tuple(x.shape)}")
    device = device or str(next(model.parameters()).device)
    roi = tuple(roi_size) if roi_size else default_roi_size(tuple(x.shape[2:]))
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=amp_device, enabled=use_amp):
            logits = sliding_window_inference(
                inputs=x.to(device),
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode=mode,
                sw_device=device,
                device=device,
            )
    return logits.float()


if __name__ == "__main__":  # pragma: no cover
    from monai.networks.nets import UNet
    net = UNet(spatial_dims=3, in_channels=1, out_channels=2,
               channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2)
    for shape in [(45, 45, 8), (160, 160, 64)]:
        xx = torch.randn(1, 1, *shape)
        out = segment_volume(net, xx, device="cpu")
        roi = default_roi_size(shape)
        print(f"{shape} roi={roi} -> {tuple(out.shape[2:])} "
              f"(full-res: {tuple(out.shape[2:]) == shape})")
