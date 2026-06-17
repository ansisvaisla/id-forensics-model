"""Download ID images from S3 using a small manifest CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from s3_zenka_ke import bucket_name, download_by_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "pilot_500.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "id_doc_front_flat"


def _read_keys(manifest_path: Path, limit: int) -> list[str]:
    keys: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("s3_key") or row.get("cache_key") or "").strip()
            if not key and row:
                key = next(iter(row.values())).strip()
            if key:
                keys.append(key)
            if len(keys) >= limit:
                break
    return keys


def _flat_filename(s3_key: str) -> str:
    relative = s3_key.removeprefix("id-doc-front/").lstrip("/")
    return relative.replace("/", "_").replace("\\", "_")


def _local_path(out_dir: Path, s3_key: str, *, flat: bool) -> Path:
    if flat:
        return out_dir / _flat_filename(s3_key)
    relative = s3_key.removeprefix("id-doc-front/").lstrip("/")
    return out_dir / relative


def _download_one(s3_key: str, out_dir: Path, bucket: str, *, flat: bool) -> tuple[str, str]:
    dest = _local_path(out_dir, s3_key, flat=flat)
    if dest.is_file():
        return s3_key, "skipped"
    download_by_key(s3_key, dest, bucket=bucket)
    return s3_key, "downloaded"


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Download images listed in a manifest CSV")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=500, help="Max rows to process")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Mirror S3 date folders instead of a single flat directory",
    )
    args = parser.parse_args()
    flat = not args.nested

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run sample_manifest.py first.", file=sys.stderr)
        return 1

    keys = _read_keys(args.manifest, args.limit)
    if not keys:
        print("No s3_key rows found in manifest.", file=sys.stderr)
        return 1

    bucket = bucket_name()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, key, args.out_dir, bucket, flat=flat): key
            for key in keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                _, status = future.result()
                if status == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"FAIL {key}: {exc}", file=sys.stderr)

    print(
        f"Done: {downloaded} downloaded, {skipped} skipped, {failed} failed "
        f"-> {args.out_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
