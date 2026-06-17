"""Pack training data for transfer to a home GPU machine.

GitHub carries code + labels JSON only. Images and YOLO splits are too large
for git — use this script to create a zip you copy via USB / cloud storage.

Usage
-----
    # After convert + split on work PC:
    python scripts/convert_labels_to_yolo.py
    python scripts/split_yolo_dataset.py

    python scripts/pack_for_home.py
    python scripts/pack_for_home.py --include-models   # also pack trained weights
    python scripts/pack_for_home.py --out D:/transfer/id_forensics_home.zip

On home PC
----------
    git clone <repo-url>
    cd id-forensics-model
    # Extract zip into repo root (merges data/yolo/ and optional models/)
    python -m venv venv && venv\\Scripts\\activate   # Windows
    pip install -r requirements.txt
    python scripts/verify_home_setup.py
    python scripts/training/train_stage2_screen.py
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths included in every pack
DATA_PATHS = [
    PROJECT_ROOT / "data" / "yolo",
    PROJECT_ROOT / "data" / "labels",
]

MODEL_GLOBS = [
    "models/stage1_corners/weights/best.pt",
    "models/stage2_screen/best.pt",
]


def _add_path(zf: zipfile.ZipFile, path: Path, arc_prefix: str = "") -> int:
    """Add file or directory tree to zip. Returns file count."""
    count = 0
    if not path.exists():
        print(f"  SKIP (missing): {path}", file=sys.stderr)
        return 0
    if path.is_file():
        arc = f"{arc_prefix}{path.name}" if arc_prefix else str(path.relative_to(PROJECT_ROOT))
        zf.write(path, arc)
        return 1
    for item in sorted(path.rglob("*")):
        if item.is_file():
            arc = str(item.relative_to(PROJECT_ROOT))
            zf.write(item, arc)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack YOLO data for home GPU transfer")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "id_forensics_home_data.zip",
        help="Output zip path (default: id_forensics_home_data.zip in repo root)",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also pack models/stage1_corners and models/stage2_screen weights",
    )
    args = parser.parse_args()

    yolo_screen = PROJECT_ROOT / "data" / "yolo" / "screen" / "images" / "train"
    if not yolo_screen.is_dir() or not any(yolo_screen.glob("*.jpg")):
        print(
            "ERROR: data/yolo/screen/images/train/ is empty or missing.\n"
            "Run first:\n"
            "  python scripts/convert_labels_to_yolo.py\n"
            "  python scripts/split_yolo_dataset.py",
            file=sys.stderr,
        )
        return 1

    total = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Packing -> {args.out}")

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in DATA_PATHS:
            n = _add_path(zf, p)
            print(f"  {p.relative_to(PROJECT_ROOT)}: {n} files")
            total += n

        if args.include_models:
            for pattern in MODEL_GLOBS:
                mp = PROJECT_ROOT / pattern
                if mp.is_file():
                    zf.write(mp, pattern)
                    total += 1
                    print(f"  {pattern}: 1 file")
                else:
                    print(f"  SKIP (missing): {pattern}", file=sys.stderr)

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Done: {total} files, {size_mb:.1f} MB")
    print(f"\nTransfer {args.out.name} to home PC and extract into the cloned repo root.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
