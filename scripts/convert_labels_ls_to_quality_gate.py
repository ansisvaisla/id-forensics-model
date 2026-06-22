"""Convert Label Studio JSON export → quality gate training dataset.

Reads a Label Studio JSON export and downloads the labelled images from S3,
writing them into the directory structure expected by train_stage2_screen.py:

    data/yolo/screen/
        images/{train,val}/<stem>.jpg
        labels/{train,val}/<stem>.txt   ← single integer class index

Usage (run from repo root):
    python scripts/convert_labels_ls_to_quality_gate.py \\
        --input data/labels/label_studio_export.json \\
        --out   data/yolo/screen \\
        --val-ratio 0.2

Class mapping (must match train_stage2_screen.py _CLASS_NAMES):
    0  screen
    1  printout
    2  selfie
    3  back
    4  garbage
    5  good_front
    6  partial
    7  blurry
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
S3_PREFIX = "id-doc-front/"

CLASS_MAP: dict[str, int] = {
    "screen": 0,
    "printout": 1,
    "selfie": 2,
    "back": 3,
    "garbage": 4,
    "good_front": 5,
    "partial": 6,
    "blurry": 7,
}

_REJECT_LABELS = {"screen", "printout", "selfie", "back", "garbage"}


def _s3_client():
    import boto3
    from botocore.config import Config
    region = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version="s3v4", max_pool_connections=32),
    )


def _download_image(client, bucket: str, s3_key: str, dest: Path) -> bool:
    """Download s3_key → dest. Returns True on success."""
    if dest.is_file():
        return True
    try:
        import io
        buf = io.BytesIO()
        client.download_fileobj(bucket, s3_key, buf)
        dest.write_bytes(buf.getvalue())
        return True
    except Exception as exc:
        print(f"  SKIP {s3_key}: {exc}", file=sys.stderr)
        return False


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key → (bucket, key)."""
    without_prefix = uri.removeprefix("s3://")
    bucket, _, key = without_prefix.partition("/")
    return bucket, key


def _flat_upload_name(file_upload: str) -> str:
    """Strip Label Studio's random upload prefix from an uploaded filename."""
    name = Path(file_upload).name
    return name.split("-", 1)[1] if "-" in name else name


def _s3_key_from_flat_upload_name(flat_name: str) -> Optional[str]:
    """Reconstruct original S3 key from YYYY_MM_DD_<filename> upload names."""
    stem, ext = flat_name.rsplit(".", 1) if "." in flat_name else (flat_name, "jpg")
    parts = stem.split("_")
    if len(parts) < 4 or not (len(parts[0]) == 4 and parts[0].isdigit()):
        return None
    year, month, day = parts[0], parts[1], parts[2]
    filename = "_".join(parts[3:])
    return f"{S3_PREFIX}{year}/{month}/{day}/{filename}.{ext}"


def _safe_stem_from_s3_key(s3_key: str) -> str:
    """Use a collision-resistant filename stem for copied training images."""
    rel = s3_key.removeprefix(S3_PREFIX).lstrip("/")
    return Path(rel.replace("/", "_")).stem


def _resolve_image_source(task: dict, default_bucket: str) -> Optional[tuple[str, str]]:
    """Return (bucket, s3_key) for S3, presigned URL, or LS-uploaded tasks."""
    image_uri: str = task.get("data", {}).get("image", "")
    if image_uri.startswith("s3://"):
        return _parse_s3_uri(image_uri)

    if image_uri.startswith("http"):
        # Presigned HTTPS URL — extract key from path component.
        from urllib.parse import urlparse
        parsed = urlparse(image_uri)
        s3_key = parsed.path.lstrip("/")
        return default_bucket, s3_key

    if image_uri.lstrip("/").startswith("data/upload/"):
        flat_name = _flat_upload_name(task.get("file_upload", "") or image_uri)
        s3_key = _s3_key_from_flat_upload_name(flat_name)
        if s3_key:
            return default_bucket, s3_key

    return None


def _extract_quality_label(task: dict) -> Optional[str]:
    """Return the quality choice from the first submitted annotation, or None."""
    for ann in task.get("annotations", []):
        if ann.get("was_cancelled") or ann.get("skipped"):
            continue
        for item in ann.get("result", []):
            if item.get("from_name") == "quality" and item.get("type") == "choices":
                choices = item.get("value", {}).get("choices", [])
                if choices:
                    return choices[0]
    return None


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Convert Label Studio export → quality gate training dataset"
    )
    parser.add_argument("--input", type=Path,
                        default=PROJECT_ROOT / "data" / "labels" / "label_studio_export.json",
                        help="Path to Label Studio JSON export")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data" / "yolo" / "screen",
                        help="Output root directory")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of each class to put in val split (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-bucket", default="sf-zenka-ke-prod-media-svc",
                        help="S3 bucket to use when image URI is not s3://")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"Loaded {len(tasks)} tasks from {args.input}")

    # ── Collect labelled samples ───────────────────────────────────────────────
    samples: list[tuple[str, str, str, int]] = []  # (bucket, s3_key, stem, class_idx)
    skipped_no_ann = skipped_bad_label = skipped_unresolved_uri = 0

    for task in tasks:
        label = _extract_quality_label(task)
        if label is None:
            skipped_no_ann += 1
            continue
        if label not in CLASS_MAP:
            skipped_bad_label += 1
            continue

        source = _resolve_image_source(task, args.default_bucket)
        if source is None:
            skipped_unresolved_uri += 1
            continue
        bucket, s3_key = source

        stem = _safe_stem_from_s3_key(s3_key)
        samples.append((bucket, s3_key, stem, CLASS_MAP[label]))

    print(f"  Annotated: {len(samples)}  |  no-annotation: {skipped_no_ann}"
          f"  |  unknown-label: {skipped_bad_label}"
          f"  |  unresolved image URI: {skipped_unresolved_uri}")
    if not samples:
        print("No samples found — nothing to do.", file=sys.stderr)
        return 1

    class_counts = Counter(idx for *_, idx in samples)
    print("Class distribution:")
    for name, idx in CLASS_MAP.items():
        print(f"  {name:12s} ({idx}): {class_counts.get(idx, 0)}")

    # ── Stratified train/val split ────────────────────────────────────────────
    random.seed(args.seed)
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[3], []).append(s)

    train_samples, val_samples = [], []
    for idx, items in by_class.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * args.val_ratio))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    print(f"\nSplit → train: {len(train_samples)}  val: {len(val_samples)}")

    # ── Create output directories ─────────────────────────────────────────────
    for child in ("images", "labels"):
        path = args.out / child
        if path.exists():
            shutil.rmtree(path)
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Download images + write label files ───────────────────────────────────
    try:
        from tqdm import tqdm  # type: ignore
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    client = _s3_client()
    failed = 0

    for split, split_samples in (("train", train_samples), ("val", val_samples)):
        img_dir = args.out / "images" / split
        lbl_dir = args.out / "labels" / split
        iterable = _tqdm(split_samples, desc=split) if _tqdm else split_samples
        for bucket, s3_key, stem, class_idx in iterable:
            dest_img = img_dir / f"{stem}.jpg"
            ok = _download_image(client, bucket, s3_key, dest_img)
            if not ok:
                failed += 1
                continue
            (lbl_dir / f"{stem}.txt").write_text(str(class_idx), encoding="utf-8")

    total = len(train_samples) + len(val_samples)
    print(f"\nDone: {total - failed}/{total} images written to {args.out}")
    print(f"  Failed downloads: {failed}")
    if failed:
        print("  Re-run to retry failed downloads (already-downloaded images are skipped).")

    print(
        "\nNext steps:\n"
        "  python scripts/training/train_stage2_screen.py --device cuda --epochs 40\n"
        "  (or run Section 4 in notebooks/colab_workbench.ipynb)"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
