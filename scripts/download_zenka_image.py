"""Download ID front or selfie images from Zenka Kenya S3."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from s3_zenka_ke import (
    ZENKA_KE_PREFIX_ID_FRONT,
    ZENKA_KE_PREFIX_SELFIE,
    bucket_name,
    download_image,
    s3_client,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def list_recent(prefix: str, limit: int) -> list[str]:
    s3 = s3_client()
    response = s3.list_objects_v2(
        Bucket=bucket_name(),
        Prefix=prefix,
        MaxKeys=limit,
    )
    return [obj["Key"] for obj in response.get("Contents", [])]


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Download Zenka KE ID images from S3")
    parser.add_argument(
        "--kind",
        choices=["id_front", "selfie"],
        help="Image folder: id-doc-front or selfie",
    )
    parser.add_argument("--key", help="S3 key or DB file path, e.g. 2023/04/12/06112ad4.jpg")
    parser.add_argument("--list", type=int, metavar="N", help="List N recent keys for --kind")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Download destination directory",
    )
    args = parser.parse_args()

    if args.list:
        if not args.kind:
            print("Use --kind id_front or --kind selfie with --list", file=sys.stderr)
            return 1
        prefix = (
            f"{ZENKA_KE_PREFIX_ID_FRONT}/"
            if args.kind == "id_front"
            else f"{ZENKA_KE_PREFIX_SELFIE}/"
        )
        keys = list_recent(prefix, args.list)
        for key in keys:
            print(key)
        return 0

    if not args.key:
        parser.error("Provide --key or use --list")

    resolved = download_image(
        args.key,
        args.out_dir / Path(args.key).name,
        kind=args.kind,
    )
    print(f"Downloaded s3://{bucket_name()}/{resolved} -> {args.out_dir / Path(args.key).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
