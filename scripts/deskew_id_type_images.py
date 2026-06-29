"""Crop id_type training images to the card bounding box for Stage 3.

Strategy: use the ML model's bounding box (not corners/warp) to crop the card.
The bounding box is much more reliable than corner precision — the model is good
at "where is the card" even when its exact keypoint placement is off.

Why not perspective warp:
  - Warp requires 4 precise corners → sensitive to small errors
  - Classical contour detection confuses the card boundary with the photo region
    or other inner rectangles, producing face crops
  - EfficientNet (Stage 3) classifies well from a bounding-box crop at a mild angle
  - Only Stage 4 (OCR) truly needs a flat, perspective-corrected image

Why bbox crop works:
  - ML bounding box encompasses the whole card reliably
  - Crops out background noise, keeps the full card visible
  - No warp = no degenerate strips, black blocks, or face crops
  - Consistent 224×224 resize happens in the training dataloader anyway

Fallback (no ML detection):
  - If the ML model is absent or fires with low confidence, the original
    image is saved with a 10% crop border trim to remove phone-camera letterboxing

Usage:
    python scripts/deskew_id_type_images.py
    python scripts/deskew_id_type_images.py --force
    python scripts/deskew_id_type_images.py --skip-missing-model
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "data" / "id_type" / "all"
DST_DIR = PROJECT_ROOT / "data" / "id_type" / "all_deskewed"
MODEL_PATH = PROJECT_ROOT / "models" / "stage2_corners" / "weights" / "best.pt"
LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "stage1_corners" / "weights" / "best.pt"

CLASSES = (
    "legacy", "maisha", "huduma", "passport",
    "driving_licence", "foreign_document", "unknown_id",
)

# Minimum ML detection confidence to trust the bounding box
_MIN_CONF = 0.35
# Padding added around the detected bbox (fraction of bbox size)
_BBOX_PAD = 0.05
# Minimum bbox area relative to image area — discard tiny detections
_MIN_BBOX_AREA_RATIO = 0.05

_ml_model = None


def _get_model():
    global _ml_model
    if _ml_model is None:
        from ultralytics import YOLO  # type: ignore
        _ml_model = YOLO(str(_model_path()))
    return _ml_model


def _model_path() -> Path:
    """Prefer new Stage 2 corner path, but keep old Stage 1 artifacts usable."""
    return MODEL_PATH if MODEL_PATH.is_file() else LEGACY_MODEL_PATH


def _bbox_crop(img, x1: float, y1: float, x2: float, y2: float,
               pad: float = _BBOX_PAD):
    """Crop image to bbox with padding, clamped to image bounds."""
    h, w = img.shape[:2]
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    cx1 = max(0, int(x1 - pw))
    cy1 = max(0, int(y1 - ph))
    cx2 = min(w, int(x2 + pw))
    cy2 = min(h, int(y2 + ph))
    return img[cy1:cy2, cx1:cx2]


def _ml_bbox_crop(img):
    """Run ML model and return bounding-box crop. Returns (crop, label)."""
    model = _get_model()
    results = model(img, verbose=False)
    if not results:
        return None, "no_detection"

    result = results[0]

    # Get bounding box from boxes (works for both pose and obb models)
    boxes = None
    conf = 0.0
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        conf = float(boxes[0].conf)
    elif hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        # OBB: get axis-aligned bounding box from xyxyxyxy
        pts = result.obb[0].xyxyxyxy[0].cpu().numpy().reshape(4, 2)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        conf = float(result.obb[0].conf)
        if conf < _MIN_CONF:
            return None, "low_confidence"
        h, w = img.shape[:2]
        area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
        if area_ratio < _MIN_BBOX_AREA_RATIO:
            return None, "bbox_too_small"
        return _bbox_crop(img, x1, y1, x2, y2), "bbox_crop"

    if boxes is None or conf < _MIN_CONF:
        return None, "low_confidence"

    # xyxy bounding box
    xyxy = boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = xyxy
    h, w = img.shape[:2]
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
    if area_ratio < _MIN_BBOX_AREA_RATIO:
        return None, "bbox_too_small"

    # Sanity: if bbox covers >90% of image, card is well-framed — return as-is
    if area_ratio > 0.90:
        return img, "full_frame"

    return _bbox_crop(img, x1, y1, x2, y2), "bbox_crop"


def _fallback_trim(img):
    """No ML: trim 8% border to remove phone letterboxing, return centre crop."""
    h, w = img.shape[:2]
    trim_y = int(h * 0.08)
    trim_x = int(w * 0.08)
    return img[trim_y:h - trim_y, trim_x:w - trim_x]


def _is_blank(img) -> bool:
    """Reject solid-colour or very dark images (no model output for these)."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < 12.0 or float(gray.mean()) < 8.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop id_type images to card bbox for Stage 3 training"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-process images even if destination already exists")
    parser.add_argument("--skip-missing-model", action="store_true",
                        help="Use trim fallback if model weights not found")
    args = parser.parse_args()

    if not SRC_DIR.is_dir():
        print(f"Source not found: {SRC_DIR}\n"
              "Run scripts/convert_labels_to_yolo.py --id-type first.", file=sys.stderr)
        return 1

    sys.path.insert(0, str(PROJECT_ROOT))

    import cv2  # type: ignore

    model_path = _model_path()
    if not model_path.is_file():
        if args.skip_missing_model:
            print(f"WARNING: model not found at {model_path}. Using trim fallback.")
            use_model = False
        else:
            print(f"ERROR: model not found: {model_path}\n"
                  "Download/restore Stage 2 corner weights first.\n"
                  "Or pass --skip-missing-model.", file=sys.stderr)
            return 1
    else:
        use_model = True

    DST_DIR.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    saved = 0
    skipped = 0

    for cls in CLASSES:
        src_cls = SRC_DIR / cls
        if not src_cls.is_dir():
            continue
        dst_cls = DST_DIR / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        images = sorted(src_cls.glob("*.jpg")) + sorted(src_cls.glob("*.jpeg"))
        for img_path in images:
            dst_path = dst_cls / f"{img_path.stem}.jpg"
            if dst_path.exists() and not args.force:
                counts["cached"] += 1
                saved += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARN: cannot read {img_path}", file=sys.stderr)
                skipped += 1
                continue

            # Reject original images that are already blank/dark
            if _is_blank(img):
                counts["blank_original"] += 1
                skipped += 1
                continue

            if use_model:
                crop, label = _ml_bbox_crop(img)
            else:
                crop, label = _fallback_trim(img), "trim_fallback"

            # If detection failed, use trim fallback rather than skip
            if crop is None:
                crop = _fallback_trim(img)
                label = f"fallback:{label}"

            # Final blank check on the crop itself
            if _is_blank(crop):
                counts["blank_crop"] += 1
                skipped += 1
                continue

            cv2.imwrite(str(dst_path), crop)
            counts[label] += 1
            saved += 1

    print(f"\nCrop complete: {saved} saved, {skipped} skipped")
    print("Outcome breakdown:")
    for lbl, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl:35s}: {n}")
    print(f"\nOutput: {DST_DIR}")
    print("Next step: python scripts/split_id_type_dataset.py --source all_deskewed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
