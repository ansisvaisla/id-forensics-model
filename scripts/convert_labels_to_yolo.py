"""Convert Label Studio polygon export to YOLOv8 Pose (keypoint) format.

Output structure:
    data/yolo/corners/
        images/all/<stem>.jpg
        labels/all/<stem>.txt        # YOLO pose: 0 cx cy w h x1 y1 2 x2 y2 2 x3 y3 2 x4 y4 2
        data.yaml                    # kpt_shape: [4, 3]

    data/yolo/screen/
        images/all/<stem>.jpg
        labels/all/<stem>.txt        # 0 (screen) or 1 (not_screen)
        data.yaml                    # classification task

Corner keypoints are the 4 polygon points from Label Studio in order:
  TL, TR, BR, BL  (the polygon order the user drew them in)

Switching from OBB to Pose lets YOLOv8 predict arbitrary quadrilaterals
(trapezoids) rather than only rotated rectangles, which is essential for
perspective-skewed ID cards.

Only completed (non-cancelled) annotations are included.
Images are resolved from data/raw/** by matching the flat filename.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_FILE = PROJECT_ROOT / "data" / "labels" / "label_studio_export.json"
YOLO_CORNERS_DIR = PROJECT_ROOT / "data" / "yolo" / "corners"
YOLO_SCREEN_DIR = PROJECT_ROOT / "data" / "yolo" / "screen"

# All directories that may contain downloaded images
RAW_ROOTS = [
    PROJECT_ROOT / "data" / "raw" / "id_doc_front_flat",
    PROJECT_ROOT / "data" / "raw" / "screen_candidates_batch2",
    PROJECT_ROOT / "data" / "raw" / "screen_candidates_batch3",
    PROJECT_ROOT / "data" / "raw" / "screen_candidates",
]
# Per-batch labeling folders (e.g. data/raw/batches/iteration2_low_liveness_750/)
RAW_BATCHES_ROOT = PROJECT_ROOT / "data" / "raw" / "batches"
# Also search nested id_doc_front tree
RAW_ROOTS_RECURSIVE = [
    PROJECT_ROOT / "data" / "raw" / "id_doc_front",
]

SCREEN_NEGATIVE_QUALITY = {"good_front", "partial", "blurry", "back"}
SCREEN_POSITIVE_QUALITY = {"screen"}

# Threshold: corners are "full-frame placeholder" if they span nearly the whole image.
# These partial images were labeled with border corners instead of real card corners.
_FULLFRAME_MIN_SPAN = 0.85  # max(xs)-min(xs) or max(ys)-min(ys) > this = bad


def _is_fullframe_polygon(pts: list[list[float]]) -> bool:
    """Return True if polygon corners span almost the entire image (bad placeholder label)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (
        min(xs) < 5
        and (max(xs) - min(xs)) / 100 > _FULLFRAME_MIN_SPAN
        and min(ys) < 10
        and (max(ys) - min(ys)) / 100 > _FULLFRAME_MIN_SPAN
    )


def _build_image_index() -> dict[str, Path]:
    """Map flat filename (e.g. '2023_05_18_7d362338.jpg') -> full path."""
    index: dict[str, Path] = {}
    for root in RAW_ROOTS:
        if not root.is_dir():
            continue
        for p in root.glob("*.jpg"):
            index[p.name] = p
        for p in root.glob("*.jpeg"):
            index[p.name] = p
    for root in RAW_ROOTS_RECURSIVE:
        if not root.is_dir():
            continue
        for p in root.rglob("*.jpg"):
            index[p.name] = p
        for p in root.rglob("*.jpeg"):
            index[p.name] = p
    if RAW_BATCHES_ROOT.is_dir():
        for p in RAW_BATCHES_ROOT.rglob("*.jpg"):
            index[p.name] = p
        for p in RAW_BATCHES_ROOT.rglob("*.jpeg"):
            index[p.name] = p
    return index


def _flat_name(file_upload: str) -> str:
    """Strip Label Studio UUID prefix: '24bedc67-2023_05_18_7d362338.jpg' -> '2023_05_18_7d362338.jpg'."""
    parts = file_upload.split("-", 1)
    return parts[1] if len(parts) == 2 else file_upload


def _parse_tasks(export_path: Path) -> list[dict]:
    return json.loads(export_path.read_text(encoding="utf-8"))


def convert_corners(tasks: list[dict], image_index: dict[str, Path], dry_run: bool = False) -> None:
    """Export 4-corner polygon labels to YOLOv8 Pose (keypoint) format.

    Label format per line:
        0 cx cy bw bh  x1 y1 2  x2 y2 2  x3 y3 2  x4 y4 2
    where cx/cy/bw/bh is the normalised bounding box of the 4 points,
    x1..x4/y1..y4 are the normalised keypoint coordinates,
    and 2 = visible.
    """
    out_images = YOLO_CORNERS_DIR / "images" / "all"
    out_labels = YOLO_CORNERS_DIR / "labels" / "all"
    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_cancelled = 0
    skipped_no_poly = 0
    skipped_no_image = 0
    skipped_fullframe = 0

    for task in tasks:
        anns = task.get("annotations") or []
        real_anns = [a for a in anns if not a.get("was_cancelled", False)]
        if not real_anns:
            skipped_cancelled += 1
            continue

        polygon = None
        for ann in real_anns:
            for r in ann.get("result", []):
                if r.get("type") == "polygonlabels":
                    pts = r.get("value", {}).get("points", [])
                    if len(pts) == 4:
                        polygon = r
                        break
            if polygon:
                break

        if polygon is None:
            skipped_no_poly += 1
            continue

        pts = polygon["value"]["points"]
        if _is_fullframe_polygon(pts):
            skipped_fullframe += 1
            continue

        flat = _flat_name(task.get("file_upload", ""))
        img_path = image_index.get(flat)
        if img_path is None:
            skipped_no_image += 1
            continue

        pts = polygon["value"]["points"]  # [[x%, y%], ...]
        # Normalise 0-1
        xs = [p[0] / 100.0 for p in pts]
        ys = [p[1] / 100.0 for p in pts]
        # Bounding box centre + size
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        # Keypoints with visibility=2 (labeled & visible)
        kpts = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in zip(xs, ys))
        label_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpts}"

        stem = Path(flat).stem
        if not dry_run:
            dest_img = out_images / flat
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
            (out_labels / f"{stem}.txt").write_text(label_line + "\n", encoding="utf-8")
        written += 1

    if not dry_run:
        _write_corners_yaml()

    print(f"Corners: {written} written, {skipped_cancelled} cancelled, "
          f"{skipped_no_poly} no polygon, {skipped_fullframe} fullframe-partial skipped, "
          f"{skipped_no_image} image not found on disk")


def convert_screen(tasks: list[dict], image_index: dict[str, Path], dry_run: bool = False) -> None:
    """Export screen classification dataset (binary: screen=0, not_screen=1)."""
    out_images = YOLO_SCREEN_DIR / "images" / "all"
    out_labels = YOLO_SCREEN_DIR / "labels" / "all"
    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

    pos = 0
    neg = 0
    skipped = 0

    for task in tasks:
        anns = task.get("annotations") or []
        real_anns = [a for a in anns if not a.get("was_cancelled", False)]
        if not real_anns:
            continue

        quality_vals: set[str] = set()
        for ann in real_anns:
            for r in ann.get("result", []):
                if r.get("type") == "choices" and r.get("from_name") == "quality":
                    quality_vals.update(r.get("value", {}).get("choices", []))

        if quality_vals & SCREEN_POSITIVE_QUALITY:
            label = 0  # screen
        elif quality_vals & SCREEN_NEGATIVE_QUALITY:
            label = 1  # not_screen (live ID)
        else:
            skipped += 1
            continue

        flat = _flat_name(task.get("file_upload", ""))
        img_path = image_index.get(flat)
        if img_path is None:
            skipped += 1
            continue

        stem = Path(flat).stem
        if not dry_run:
            dest_img = out_images / flat
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
            (out_labels / f"{stem}.txt").write_text(f"{label}\n", encoding="utf-8")

        if label == 0:
            pos += 1
        else:
            neg += 1

    if not dry_run:
        _write_screen_yaml()

    print(f"Screen: {pos} positives (screen), {neg} negatives (live), {skipped} skipped")


def _write_corners_yaml() -> None:
    yaml_content = (
        "path: ../data/yolo/corners\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "kpt_shape: [4, 3]  # 4 keypoints, each (x, y, visibility)\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: id_card\n"
    )
    (YOLO_CORNERS_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")


def _write_screen_yaml() -> None:
    yaml_content = (
        "path: ../data/yolo/screen\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: screen\n"
        "  1: live\n"
        "\n"
        "task: classify\n"
    )
    (YOLO_SCREEN_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Label Studio export to YOLOv8 format")
    parser.add_argument("--export", type=Path, default=EXPORT_FILE)
    parser.add_argument("--corners", action="store_true", default=True, help="Convert corner polygons")
    parser.add_argument("--screen", action="store_true", default=True, help="Convert screen labels")
    parser.add_argument("--no-corners", dest="corners", action="store_false")
    parser.add_argument("--no-screen", dest="screen", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.export.is_file():
        print(f"Export not found: {args.export}", file=sys.stderr)
        return 1

    print(f"Building image index from {len(RAW_ROOTS) + len(RAW_ROOTS_RECURSIVE)} root(s)...")
    image_index = _build_image_index()
    print(f"  {len(image_index)} images indexed")

    tasks = _parse_tasks(args.export)
    print(f"  {len(tasks)} tasks in export")

    if args.corners:
        convert_corners(tasks, image_index, dry_run=args.dry_run)
    if args.screen:
        convert_screen(tasks, image_index, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run — no files written.")
    else:
        print(f"\nCorners output: {YOLO_CORNERS_DIR}")
        print(f"Screen output:  {YOLO_SCREEN_DIR}")
        print("\nNext step: run scripts/split_yolo_dataset.py to create train/val/test splits.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
