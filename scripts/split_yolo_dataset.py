"""Split YOLOv8 dataset (corners or screen) into train/val/test.

For corners: image-level random split 70/15/15 (no client_id available in export).
For screen: stratified split by class label to keep class ratio balanced.

Moves images+labels from images/all and labels/all into images/{split} and labels/{split}.
Idempotent — safe to re-run; skips already-split files.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = PROJECT_ROOT / "data" / "yolo"

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _read_label(label_file: Path) -> str | None:
    """Read first token of first line as class label."""
    try:
        return label_file.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        return None


def split_dataset(dataset_dir: Path, seed: int, stratify: bool) -> None:
    all_images = dataset_dir / "images" / "all"
    all_labels = dataset_dir / "labels" / "all"

    if not all_images.is_dir():
        print(f"No 'images/all' found in {dataset_dir}. Already split?", file=sys.stderr)
        return

    stems = [p.stem for p in sorted(all_images.glob("*.jpg"))]
    if not stems:
        print(f"No images found in {all_images}", file=sys.stderr)
        return

    print(f"\nDataset: {dataset_dir.name}  ({len(stems)} images)")

    if stratify:
        # Group by class label
        groups: dict[str, list[str]] = defaultdict(list)
        for stem in stems:
            lbl = _read_label(all_labels / f"{stem}.txt") or "unknown"
            groups[lbl].append(stem)
        for cls, items in groups.items():
            print(f"  class {cls}: {len(items)}")
        buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for cls_stems in groups.values():
            rng = random.Random(seed)
            rng.shuffle(cls_stems)
            n = len(cls_stems)
            n_val = max(1, round(n * SPLITS["val"]))
            n_test = max(1, round(n * SPLITS["test"]))
            buckets["val"].extend(cls_stems[:n_val])
            buckets["test"].extend(cls_stems[n_val : n_val + n_test])
            buckets["train"].extend(cls_stems[n_val + n_test :])
    else:
        rng = random.Random(seed)
        shuffled = stems[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, round(n * SPLITS["val"]))
        n_test = max(1, round(n * SPLITS["test"]))
        buckets = {
            "val": shuffled[:n_val],
            "test": shuffled[n_val : n_val + n_test],
            "train": shuffled[n_val + n_test :],
        }

    for split, split_stems in buckets.items():
        img_out = dataset_dir / "images" / split
        lbl_out = dataset_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        moved = 0
        for stem in split_stems:
            src_img = all_images / f"{stem}.jpg"
            src_lbl = all_labels / f"{stem}.txt"
            dst_img = img_out / f"{stem}.jpg"
            dst_lbl = lbl_out / f"{stem}.txt"
            if not dst_img.exists() and src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)  # always overwrite — labels may have changed format
            moved += 1
        print(f"  {split:5s}: {moved}")

    # Write split manifest
    manifest = {split: stems for split, stems in buckets.items()}
    manifest_path = dataset_dir / "splits.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test")
    parser.add_argument(
        "--dataset",
        choices=["corners", "screen", "both"],
        default="both",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets: list[tuple[Path, bool]] = []
    if args.dataset in ("corners", "both"):
        datasets.append((YOLO_DIR / "corners", False))
    if args.dataset in ("screen", "both"):
        datasets.append((YOLO_DIR / "screen", True))

    for ds_path, stratify in datasets:
        if not ds_path.is_dir():
            print(f"Dataset not found: {ds_path}. Run convert_labels_to_yolo.py first.", file=sys.stderr)
            continue
        split_dataset(ds_path, seed=args.seed, stratify=stratify)

    return 0


if __name__ == "__main__":
    sys.exit(main())
