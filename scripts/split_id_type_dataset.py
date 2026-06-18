"""Stratified train/val/test split for the ID type ImageFolder dataset.

Reads images from data/id_type/all/<class>/ (raw) or
data/id_type/all_deskewed/<class>/ (Stage-1 pre-processed, recommended)
and copies them into data/id_type/{train,val,test}/<class>/.

Split ratios: 70% train / 15% val / 15% test, stratified per class.
Classes with <5 images are put entirely in train.

Usage:
    python scripts/split_id_type_dataset.py                         # raw images
    python scripts/split_id_type_dataset.py --source all_deskewed   # after deskew step
    python scripts/split_id_type_dataset.py --seed 123
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "id_type"

CLASSES = (
    "legacy",
    "maisha",
    "huduma",
    "passport",
    "driving_licence",
    "foreign_document",
    "unknown_id",
)
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _discover_samples(source_dir: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image paths]} from given source dir."""
    per_class: dict[str, list[Path]] = defaultdict(list)
    for cls in CLASSES:
        cls_dir = source_dir / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            per_class[cls].append(p)
        for p in sorted(cls_dir.glob("*.jpeg")):
            per_class[cls].append(p)
    return dict(per_class)


def _split_class(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    """Stratified split for a single class."""
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n = len(paths)

    if n < 5:
        # Too small to split meaningfully — all go to train
        return {"train": paths, "val": [], "test": []}

    n_val = max(1, round(n * SPLITS["val"]))
    n_test = max(1, round(n * SPLITS["test"]))
    n_train = n - n_val - n_test

    return {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split id_type dataset into train/val/test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--source",
        default="all",
        help="Source subfolder under data/id_type/ (default: 'all'; use 'all_deskewed' after deskew step)",
    )
    args = parser.parse_args()

    source_dir = DATA_DIR / args.source
    if not source_dir.is_dir():
        msg = f"Source directory not found: {source_dir}\n"
        if args.source == "all_deskewed":
            msg += "Run scripts/deskew_id_type_images.py first."
        else:
            msg += "Run scripts/convert_labels_to_yolo.py --id-type first."
        print(msg, file=sys.stderr)
        return 1

    per_class = _discover_samples(source_dir)
    if not per_class:
        print("No images found in data/id_type/all/ — check convert step.", file=sys.stderr)
        return 1

    total_written = 0

    for cls, paths in per_class.items():
        splits = _split_class(paths, seed=args.seed)
        for split_name, split_paths in splits.items():
            dest_dir = DATA_DIR / split_name / cls
            if not args.dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                dst = dest_dir / src.name
                if not args.dry_run:
                    shutil.copy2(src, dst)
                total_written += 1

        counts = {s: len(p) for s, p in splits.items()}
        print(f"  {cls:20s}  total={len(paths):3d}  "
              f"train={counts['train']:3d}  val={counts['val']:3d}  test={counts['test']:3d}")

    if args.dry_run:
        print(f"\nDry run — would write {total_written} files.")
    else:
        # Write manifest split
        manifest_path = DATA_DIR / "splits.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stem", "class", "split"])
            for cls, paths in per_class.items():
                splits = _split_class(paths, seed=args.seed)
                for split_name, split_paths in splits.items():
                    for p in split_paths:
                        writer.writerow([p.stem, cls, split_name])
        print(f"\n{total_written} files written -> {DATA_DIR}")
        if args.source == "all_deskewed":
            print("Next step: python scripts/training/train_stage4_id_type.py --device cuda")
        else:
            print("TIP: run scripts/deskew_id_type_images.py first for better Stage 4 accuracy,")
            print("     then re-run with --source all_deskewed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
