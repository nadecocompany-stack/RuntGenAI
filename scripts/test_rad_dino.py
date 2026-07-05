"""
test_rad_dino.py
==================================================================
Validates the RAD-DINO CXR wiring WITHOUT downloading hub weights.

Part A — production geometry: build the real DINOv2 architecture at RAD-DINO's
         518x518 / patch-14 setting and confirm the token layout the model card
         specifies: last_hidden_state (1, 1370, 768) = 1 CLS + 37*37 patches,
         and that the patch grid folds to (768, 37, 37).
Part B — localization path: at a reduced 224 size (fast on CPU) run the full
         forward -> Grad-CAM -> boxes chain and confirm boxes are produced.

A PASS means the wiring is correct; swap ``pretrained=False`` for
``pretrained=True`` (with hub access) to run the same code on real weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cxr_rad_dino import (                                    # noqa: E402
    build_rad_dino_cxr, localize_cxr, RAD_DINO_HIDDEN,
)
from explainability import (                                  # noqa: E402
    compute_gradcam, make_vit_reshape_transform,
)


def part_a_geometry() -> bool:
    print("-- Part A: RAD-DINO production geometry (518 / patch 14) --")
    model = build_rad_dino_cxr(pretrained=False, image_size=518, device="cpu")
    x = torch.randn(1, 3, 518, 518)
    with torch.no_grad():
        hs = model.backbone(pixel_values=x).last_hidden_state
    n_tokens = hs.shape[1]
    grid = int(round((n_tokens - model.num_prefix_tokens) ** 0.5))
    reshape = make_vit_reshape_transform(model.num_prefix_tokens)
    folded = reshape(hs)

    ok = (
        hs.shape == (1, 1370, RAD_DINO_HIDDEN)
        and model.num_prefix_tokens == 1
        and grid == 37
        and tuple(folded.shape) == (1, RAD_DINO_HIDDEN, 37, 37)
    )
    print(f"   last_hidden_state = {tuple(hs.shape)}  (expect (1, 1370, 768))")
    print(f"   prefix tokens     = {model.num_prefix_tokens}  (expect 1: CLS, no registers)")
    print(f"   patch grid        = {grid}x{grid}  (expect 37x37)")
    print(f"   folded patches    = {tuple(folded.shape)}  (expect (1, 768, 37, 37))")
    print(f"   => {'PASS' if ok else 'FAIL'}")
    return ok


def part_b_localization() -> bool:
    print("-- Part B: forward -> Grad-CAM -> boxes (reduced 224 for speed) --")
    model = build_rad_dino_cxr(pretrained=False, image_size=224, device="cpu")
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)

    reshape = make_vit_reshape_transform(model.num_prefix_tokens)
    peaks = [(c, float(compute_gradcam(model, x, model.gradcam_target_layer, c,
                                       multilabel=True, reshape_transform=reshape).max()))
             for c in range(1, 14)]
    target = max(peaks, key=lambda t: t[1])[0]
    boxes = localize_cxr(model, x, target, min_region_pixels=2)

    ok = (tuple(logits.shape) == (1, 14)) and len(boxes) > 0
    print(f"   logits            = {tuple(logits.shape)}  (expect (1, 14))")
    print(f"   strongest finding = class {target}")
    print(f"   boxes             = {len(boxes)}")
    for b in boxes:
        print(f"     class={b.class_id} peak={b.peak_confidence:.2f} "
              f"patches={b.voxel_count} bbox(y0,y1,x0,x1)={b.bbox}")
    print(f"   => {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("=" * 66)
    print("RAD-DINO CXR WIRING TEST  (offline random weights)")
    print("=" * 66)
    a = part_a_geometry()
    b = part_b_localization()
    print("=" * 66)
    verdict = "ALL GREEN" if (a and b) else "FAILURE"
    print(f"RESULT: {verdict}")
    print("=" * 66)
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
