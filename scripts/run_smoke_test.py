"""
run_smoke_test.py
==================================================================
End-to-end smoke test on RANDOMLY-INITIALIZED models.

Pipeline exercised per case:
    synthetic DICOM series
      -> load_and_preprocess   (pydicom + MONAI: metadata, HU/normalise, tensor)
      -> route_to_model        (modality/body-part -> registry key)
      -> ModelRegistry.build   (correct in/out channels, random weights)
      -> forward pass          (logits)
      -> localize:
           MASK_CC  keys -> ConfidenceLocalizer (segmentation logits -> boxes)
           GRADCAM  keys -> compute_gradcam -> heatmap_to_boxes

Outputs are meaningless (random weights) — a PASS means every stage ran and
produced correctly-shaped, well-formed results. This is a plumbing test.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import torch

# Make src/ importable when run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dicom_ingestion import load_and_preprocess, route_to_model     # noqa: E402
from label_taxonomies import get_spec, Localization                 # noqa: E402
from model_registry import ModelRegistry, _in_channels              # noqa: E402
from confidence_localization import ConfidenceLocalizer             # noqa: E402
from explainability import compute_gradcam, heatmap_to_boxes        # noqa: E402
from synthetic_dicom import generate_all                            # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s | %(name)s | %(message)s")


def _match_channels(tensor: torch.Tensor, want: int) -> torch.Tensor:
    """
    Smoke-test helper: BraTS expects 4 co-registered MR sequences as channels.
    Ingestion yields 1 channel from a single series, so we tile it to `want`
    channels purely to exercise the shape. Real pipelines stack the actual
    T1/T1Gd/T2/FLAIR volumes here.
    """
    have = tensor.shape[0]
    if have == want:
        return tensor
    if want % have == 0:
        return tensor.repeat(want // have, *([1] * (tensor.ndim - 1)))
    return tensor[:1].repeat(want, *([1] * (tensor.ndim - 1)))


def run() -> int:
    registry = ModelRegistry(device="cpu", seed=0)
    tmp = tempfile.mkdtemp(prefix="smoke_")
    series = generate_all(tmp)
    failures = 0

    print("=" * 74)
    print("RADIOLOGY PIPELINE SMOKE TEST  (random weights — plumbing only)")
    print("=" * 74)

    for case, dicom_dir in series.items():
        try:
            # --- ingest -------------------------------------------------
            scan = load_and_preprocess(dicom_dir)
            key = route_to_model(scan.metadata)
            spec = get_spec(key)
            tag = f"{case:9s} | {scan.metadata.modality.value}/{scan.metadata.body_part}"
            if spec is None:
                print(f"[SKIP] {tag} -> key '{key}' has no ModelSpec")
                continue

            # --- build model + shape the input --------------------------
            model = registry.build(key)
            x = scan.tensor
            x = _match_channels(x, _in_channels(spec)).unsqueeze(0)  # add batch

            # --- infer + localize --------------------------------------
            if spec.localization == Localization.MASK_CC:
                # Sliding-window inference handles any volume size and returns
                # full-resolution logits (no DivisiblePad, no pad offset).
                from coordinate_mapping import map_detection_bbox
                from inference import segment_volume, default_roi_size
                logits = segment_volume(model, x, overlap=0.25, mode="gaussian")[0]
                loc = ConfidenceLocalizer(
                    mode="sigmoid" if spec.multilabel else "softmax",
                    prob_threshold=0.5, min_region_voxels=5,
                )
                result = loc(logits)
                loc_desc = (f"roi={default_roi_size(tuple(x.shape[2:]))} "
                            f"top_cls={result.top_class} "
                            f"boxes={len(result.detections)}")
                if result.detections:
                    m = map_detection_bbox(
                        result.detections[0].bbox,
                        preprocessed_affine=scan.preprocessed_affine,
                        original_affine=scan.original_affine,
                        clip_shape=scan.original_shape)
                    loc_desc += f" | orig_voxel={m['original_voxel']}"
            else:  # GRADCAM classification path
                from coordinate_mapping import map_detection_bbox
                pre_shape = tuple(int(s) for s in x.shape[2:])
                logits = model(x)                  # (1, C)
                probs = logits.sigmoid()[0] if spec.multilabel else logits.softmax(1)[0]
                # Pick the highest-scoring *finding* class (skip index 0:
                # 'any'/'No Finding') to localise.
                finding_scores = probs.clone(); finding_scores[0] = -1
                target = int(finding_scores.argmax())
                cam = compute_gradcam(model, x, model.feature_layer, target,
                                      multilabel=spec.multilabel)
                boxes = heatmap_to_boxes(cam, class_id=target,
                                         rel_threshold=0.5, min_region_voxels=2)
                loc_desc = (f"target='{spec.class_map[target]}' "
                            f"p={float(probs[target]):.2f} "
                            f"cam{cam.shape} boxes={len(boxes)}")
                if boxes:
                    m = map_detection_bbox(
                        boxes[0].bbox, feature_shape=cam.shape,
                        input_shape=pre_shape,
                        preprocessed_affine=scan.preprocessed_affine,
                        original_affine=scan.original_affine,
                        clip_shape=scan.original_shape)
                    loc_desc += f" | orig_voxel={m['original_voxel']}"

            print(f"[PASS] {tag:26s} -> {key:26s} "
                  f"in{tuple(x.shape)} | {loc_desc}")

        except Exception as exc:  # noqa: BLE001 — smoke test wants the reason
            failures += 1
            print(f"[FAIL] {case:9s} -> {type(exc).__name__}: {exc}")
            logging.exception("case %s failed", case)

    print("=" * 74)
    verdict = "ALL STAGES GREEN" if failures == 0 else f"{failures} FAILURE(S)"
    print(f"RESULT: {verdict}  ({len(series) - failures}/{len(series)} cases passed)")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
