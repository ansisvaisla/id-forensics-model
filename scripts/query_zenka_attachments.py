"""Query Zenka Kenya ID attachments from postgres2 and optionally download from S3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from db_zenka_ke import (
    SUB_TYPE_ID_FRONT,
    SUB_TYPE_SELFIE,
    ZenkaAttachmentRow,
    fetch_attachments_with_s3_keys,
)
from s3_zenka_ke import download_by_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUB_TYPE_CHOICES = {
    "id_front": SUB_TYPE_ID_FRONT,
    "selfie": SUB_TYPE_SELFIE,
}


def _row_to_dict(row: ZenkaAttachmentRow) -> dict:
    return asdict(row)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Query Zenka KE attachments with S3 keys (postgres2 + S3)"
    )
    parser.add_argument(
        "--kind",
        choices=["id_front", "selfie", "both"],
        default="both",
        help="Which images to fetch",
    )
    parser.add_argument("--limit", type=int, default=5, help="Rows per kind")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Look back window on integration.request (keeps queries fast)",
    )
    parser.add_argument("--download", action="store_true", help="Download images to ./data/")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Download destination when using --download",
    )
    args = parser.parse_args()

    kinds = list(SUB_TYPE_CHOICES) if args.kind == "both" else [args.kind]
    all_rows: list[ZenkaAttachmentRow] = []

    for kind in kinds:
        sub_type = SUB_TYPE_CHOICES[kind]
        rows = fetch_attachments_with_s3_keys(
            sub_type,
            limit=args.limit,
            hours=args.hours,
            default_bucket=os.getenv("AWS_S3_BUCKET", "sf-zenka-ke-prod-media-svc"),
        )
        all_rows.extend(rows)

    print(json.dumps([_row_to_dict(row) for row in all_rows], default=str, indent=2))

    if not args.download:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for row in all_rows:
        filename = Path(row.s3_key).name
        dest = args.out_dir / f"{row.sub_type.lower()}_{row.attachment_id}_{filename}"
        download_by_key(row.s3_key, dest, bucket=row.bucket)
        print(f"Downloaded s3://{row.bucket}/{row.s3_key} -> {dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
