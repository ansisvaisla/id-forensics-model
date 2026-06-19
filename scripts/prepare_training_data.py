"""Download labeled training images from S3 and build YOLO splits.

Work PC (has .env AWS credentials):
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --write-colab-manifest

Colab (no AWS credentials — uses a small pre-signed URL manifest):
    python scripts/prepare_training_data.py --from-colab-manifest /content/colab_manifest.json

Images are resolved from Label Studio export flat filenames, e.g.
  2023_11_22_d1171082.jpg  ->  s3://bucket/id-doc-front/2023/11/22/d1171082.jpg

Training scripts then read:
  data/yolo/screen/images/{train,val,test}/
  data/yolo/corners/images/{train,val,test}/
  data/id_type/{train,val,test}/   (after deskew + split)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_FILE = PROJECT_ROOT / "data" / "labels" / "label_studio_export.json"
RAW_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "id_doc_front_flat"
DEFAULT_COLAB_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "colab_presigned.json"


def _flat_name(file_upload: str) -> str:
    parts = file_upload.split("-", 1)
    return parts[1] if len(parts) == 2 else file_upload


def flat_to_s3_key(flat_name: str) -> str:
    """Reverse the flat filename convention used by download_from_manifest.py."""
    path = Path(flat_name)
    stem = path.stem
    ext = path.suffix or ".jpg"
    parts = stem.split("_")
    if len(parts) >= 4 and len(parts[0]) == 4 and parts[0].isdigit():
        year, month, day = parts[0], parts[1], parts[2]
        fname = "_".join(parts[3:])
        return f"id-doc-front/{year}/{month}/{day}/{fname}{ext}"
    return f"id-doc-front/{stem}{ext}"


def labeled_flat_names(export_path: Path) -> list[str]:
    """Return unique flat image filenames referenced in the Label Studio export."""
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    names: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        anns = task.get("annotations") or []
        if not any(not a.get("was_cancelled", False) for a in anns):
            continue
        flat = _flat_name(task.get("file_upload", ""))
        if flat and flat not in seen:
            seen.add(flat)
            names.append(flat)
    return names


def _run_pipeline_steps(skip_id_type_split: bool) -> None:
    steps = [
        [sys.executable, "scripts/convert_labels_to_yolo.py"],
        [sys.executable, "scripts/split_yolo_dataset.py", "--dataset", "both"],
    ]
    if not skip_id_type_split:
        steps.append(
            [sys.executable, "scripts/split_id_type_dataset.py", "--source", "all"]
        )
    for cmd in steps:
        print(f"\n>> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _download_from_s3(
    flat_names: list[str],
    out_dir: Path,
    *,
    workers: int,
) -> tuple[int, int, int]:
    from s3_zenka_ke import bucket_name, download_by_key

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0

    def _one(flat: str) -> tuple[str, str]:
        dest = out_dir / flat
        if dest.is_file():
            return flat, "skipped"
        key = flat_to_s3_key(flat)
        download_by_key(key, dest, bucket=bucket_name())
        return flat, "downloaded"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, flat): flat for flat in flat_names}
        for future in as_completed(futures):
            flat = futures[future]
            try:
                _, status = future.result()
                if status == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"FAIL {flat}: {exc}", file=sys.stderr)

    return downloaded, skipped, failed


def _write_colab_manifest(flat_names: list[str], manifest_path: Path, expires: int) -> None:
    import boto3
    from s3_zenka_ke import bucket_name, s3_client

    client = s3_client()
    bucket = bucket_name()
    entries: list[dict[str, str]] = []

    for flat in flat_names:
        key = flat_to_s3_key(flat)
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
        entries.append({"flat_name": flat, "s3_key": key, "url": url})

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bucket": bucket,
        "expires_seconds": expires,
        "count": len(entries),
        "images": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    size_kb = manifest_path.stat().st_size / 1024
    print(f"Colab manifest: {manifest_path} ({len(entries)} images, {size_kb:.0f} KB)")


def _download_from_colab_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    workers: int,
) -> tuple[int, int, int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = payload.get("images") or []
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0

    def _one(entry: dict) -> tuple[str, str]:
        flat = entry["flat_name"]
        dest = out_dir / flat
        if dest.is_file():
            return flat, "skipped"
        urlretrieve(entry["url"], dest)
        return flat, "downloaded"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, entry): entry for entry in images}
        for future in as_completed(futures):
            entry = futures[future]
            flat = entry.get("flat_name", "?")
            try:
                _, status = future.result()
                if status == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"FAIL {flat}: {exc}", file=sys.stderr)

    return downloaded, skipped, failed


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Download labeled images and build train/val/test splits"
    )
    parser.add_argument(
        "--from-colab-manifest",
        type=Path,
        help="Colab mode: download images via pre-signed HTTPS URLs (no AWS creds)",
    )
    parser.add_argument(
        "--write-colab-manifest",
        action="store_true",
        help="Work PC: write a small JSON manifest with pre-signed URLs for Colab",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_COLAB_MANIFEST,
        help="Output path for --write-colab-manifest",
    )
    parser.add_argument(
        "--manifest-expires",
        type=int,
        default=86_400,
        help="Pre-signed URL lifetime in seconds (default 24h)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RAW_OUT_DIR,
        help="Where to save raw JPEGs",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only run convert + split (images already on disk)",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Only download images; do not run convert/split",
    )
    parser.add_argument(
        "--skip-id-type-split",
        action="store_true",
        help="Skip id_type split (use when all_deskewed not ready yet)",
    )
    args = parser.parse_args()

    if not EXPORT_FILE.is_file():
        print(f"Label export not found: {EXPORT_FILE}", file=sys.stderr)
        return 1

    flat_names = labeled_flat_names(EXPORT_FILE)
    print(f"Labeled images in export: {len(flat_names)}")

    if args.from_colab_manifest:
        if not args.from_colab_manifest.is_file():
            print(f"Manifest not found: {args.from_colab_manifest}", file=sys.stderr)
            return 1
        dl, sk, fail = _download_from_colab_manifest(
            args.from_colab_manifest, args.out_dir, workers=args.workers
        )
        print(f"Colab download: {dl} new, {sk} cached, {fail} failed -> {args.out_dir}")
        if fail:
            return 1
    elif not args.skip_download:
        dl, sk, fail = _download_from_s3(flat_names, args.out_dir, workers=args.workers)
        print(f"S3 download: {dl} new, {sk} cached, {fail} failed -> {args.out_dir}")
        if fail:
            return 1

    if args.write_colab_manifest:
        _write_colab_manifest(flat_names, args.manifest_out, args.manifest_expires)
        print(
            "Upload this one small file to Colab (Files panel or Drive):\n"
            f"  {args.manifest_out}\n"
            "Then in Colab run:\n"
            f"  python scripts/prepare_training_data.py "
            f"--from-colab-manifest /content/colab_presigned.json"
        )

    if not args.skip_pipeline:
        _run_pipeline_steps(skip_id_type_split=args.skip_id_type_split)

    return 0


if __name__ == "__main__":
    sys.exit(main())
