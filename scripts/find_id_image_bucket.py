"""Find which accessible S3 bucket contains ERP client ID document images.

Use a real key from `erp_finbro_ph.client_id_document.file`, for example:

    python scripts/find_id_image_bucket.py --key 2026/06/04/client_id_front_x.jpg

The script checks each visible bucket with `HeadObject`. A 404 means the key is
not there. A 403 means the bucket/key may exist but your role lacks permission.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProbeResult:
    bucket: str
    key: str
    status: str
    detail: str


def build_session() -> boto3.Session:
    load_dotenv(PROJECT_ROOT / ".env")
    region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
    profile = os.getenv("AWS_PROFILE")

    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


def list_buckets(session: boto3.Session) -> list[str]:
    s3 = session.client("s3")
    response = s3.list_buckets()
    return sorted(bucket["Name"] for bucket in response.get("Buckets", []))


def candidate_keys(raw_key: str, extra_prefixes: list[str]) -> list[str]:
    key = raw_key.lstrip("/")
    prefixes = [
        "",
        "PH/",
        "ph/",
        "uploads/",
        "upload/",
        "documents/",
        "client_id_document/",
        "client_id_documents/",
        "id_documents/",
        "private/",
        "public/",
        *extra_prefixes,
    ]

    candidates: list[str] = []
    for prefix in prefixes:
        normalized_prefix = prefix.strip("/")
        candidate = f"{normalized_prefix}/{key}" if normalized_prefix else key
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def probe_key(session: boto3.Session, bucket: str, key: str) -> ProbeResult:
    s3 = session.client("s3")
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        if code in {"403", "AccessDenied"}:
            return ProbeResult(bucket, key, "FORBIDDEN", "No HeadObject/GetObject permission")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return ProbeResult(bucket, key, "MISSING", "Object not found")
        if code == "NoSuchBucket":
            return ProbeResult(bucket, key, "NO_BUCKET", "Bucket does not exist or is not visible")
        return ProbeResult(bucket, key, "ERROR", f"{code}: {exc}")

    size = response.get("ContentLength", 0)
    modified = response.get("LastModified")
    return ProbeResult(bucket, key, "FOUND", f"{size} bytes, last modified {modified}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe S3 buckets for raw ID document images")
    parser.add_argument("--key", action="append", required=True, help="S3 key from client_id_document.file")
    parser.add_argument("--bucket", action="append", help="Specific bucket to test; default tests all visible")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Extra prefix candidate to prepend before the DB key, e.g. ERP/PH",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print missing checks too. By default only FOUND/FORBIDDEN/ERROR is shown.",
    )
    args = parser.parse_args()

    try:
        session = build_session()
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        buckets = args.bucket or list_buckets(session)
    except ProfileNotFound as exc:
        print(f"Profile not found: {exc}", file=sys.stderr)
        return 1
    except NoCredentialsError:
        print("No AWS credentials found. Fill .env first.", file=sys.stderr)
        return 1
    except ClientError as exc:
        print(f"AWS error: {exc}", file=sys.stderr)
        return 1

    print("Authenticated as:")
    print(f"  Account: {identity['Account']}")
    print(f"  ARN:     {identity['Arn']}")
    print(f"\nTesting {len(buckets)} bucket(s) and {len(args.key)} DB key(s).\n")

    found_any = False
    for bucket in buckets:
        for raw_key in args.key:
            for key in candidate_keys(raw_key, args.prefix):
                result = probe_key(session, bucket, key)
                if result.status == "FOUND":
                    found_any = True
                    print(f"FOUND     s3://{result.bucket}/{result.key} ({result.detail})")
                elif result.status in {"FORBIDDEN", "ERROR"}:
                    print(f"{result.status:<9} s3://{result.bucket}/{result.key} ({result.detail})")
                elif args.show_missing:
                    print(f"MISSING   s3://{result.bucket}/{result.key}")

    if not found_any:
        print(
            "\nNo accessible object was found. If you saw FORBIDDEN, ask for s3:GetObject "
            "on that bucket/prefix. If everything was MISSING, the raw images are likely "
            "in a different ERP/application AWS account or bucket."
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
