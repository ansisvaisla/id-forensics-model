"""Train Stage 1 — YOLOv8-Pose corner keypoint detector.

Trains a pose estimation model to predict 4 keypoints (card corners)
rather than a rotated bounding box (OBB). This correctly handles
perspective-skewed cards which are trapezoidal, not rectangular.

Usage:
    python scripts/training/train_stage1_corners.py
    python scripts/training/train_stage1_corners.py --model yolov8s-pose.pt --epochs 100

Output: models/stage1_corners/  (weights/best.pt, weights/last.pt)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "corners" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models" / "stage1_corners"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8-Pose corner keypoint detector")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Base pose model checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu', '0', '0,1', etc.")
    args = parser.parse_args()

    if not DATA_YAML.is_file():
        print(f"data.yaml not found: {DATA_YAML}", file=sys.stderr)
        print("Run scripts/convert_labels_to_yolo.py first.", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        return 1

    # Wipe stale label caches — required when switching task type (OBB -> Pose)
    # or re-running after convert_labels_to_yolo.py regenerates labels.
    labels_root = DATA_YAML.parent / "labels"
    for cache_file in labels_root.rglob("*.cache"):
        cache_file.unlink()
        print(f"  deleted stale cache: {cache_file.relative_to(PROJECT_ROOT)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training YOLOv8-Pose on {DATA_YAML}")
    print(f"  base model : {args.model}")
    print(f"  epochs     : {args.epochs}")
    print(f"  batch      : {args.batch}")
    print(f"  device     : {args.device}")
    print(f"  output     : {OUTPUT_DIR}")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        cache="ram",   # cache decoded images in RAM — eliminates Drive FUSE reads after first epoch
        workers=4,     # parallel image loading
    )

    best = OUTPUT_DIR / "weights" / "best.pt"
    if best.is_file():
        print(f"\nBest weights: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
