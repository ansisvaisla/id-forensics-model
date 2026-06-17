"""List and sample-read objects from the PH S3 feature_store bucket."""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _client():
    load_dotenv(PROJECT_ROOT / ".env")
    region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
    profile = os.getenv("AWS_PROFILE")
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.Session(region_name=region)
    return session.client("s3")


def list_prefixes(s3, bucket: str, prefix: str, max_keys: int) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    seen: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            seen.add(entry["Prefix"])
        for obj in page.get("Contents", []):
            seen.add(obj["Key"])
        if len(seen) >= max_keys:
            break
    return sorted(seen)[:max_keys]


def list_files(s3, bucket: str, prefix: str, max_keys: int) -> list[dict]:
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    return [
        {
            "key": obj["Key"],
            "size_kb": round(obj["Size"] / 1024, 1),
            "modified": obj["LastModified"].isoformat(),
        }
        for obj in response.get("Contents", [])
        if not obj["Key"].endswith("/")
    ]


def read_parquet_head(s3, bucket: str, key: str, rows: int) -> None:
    import pandas as pd

    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    print(f"\nParquet: s3://{bucket}/{key}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:20]}")
    print(df.head(rows).to_string())


def read_csv_head(s3, bucket: str, key: str, rows: int) -> None:
    import pandas as pd

    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"], nrows=rows)
    print(f"\nCSV: s3://{bucket}/{key}")
    print(f"Columns: {list(df.columns)}")
    print(df.to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore PH S3 feature_store")
    parser.add_argument("--bucket", default=os.getenv("AWS_S3_BUCKET"))
    parser.add_argument("--prefix", default=os.getenv("AWS_S3_PREFIX", "feature_store/"))
    parser.add_argument("--dataset", help="Subfolder under prefix, e.g. cibi_personal_info_features")
    parser.add_argument("--max-keys", type=int, default=30)
    parser.add_argument("--peek", action="store_true", help="Read first data file found")
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not args.bucket:
        print("Set AWS_S3_BUCKET in .env (the PH bucket name from the console URL).", file=sys.stderr)
        return 1

    prefix = args.prefix
    if args.dataset:
        prefix = f"{prefix.rstrip('/')}/{args.dataset.strip('/')}/"

    s3 = _client()
    print(f"Bucket: {args.bucket}")
    print(f"Prefix: {prefix}")

    try:
        entries = list_prefixes(s3, args.bucket, prefix, args.max_keys)
    except ClientError as exc:
        print(f"S3 error: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("No objects found under this prefix.")
        return 0

    print(f"\nFound {len(entries)} entries:")
    for entry in entries:
        print(f"  {entry}")

    if args.peek:
        files = list_files(s3, args.bucket, prefix, max_keys=50)
        data_files = [
            f for f in files
            if f["key"].endswith((".parquet", ".csv", ".json", ".snappy.parquet"))
        ]
        if not data_files:
            print("\nNo parquet/csv/json files in first 50 objects. Drill into a subfolder:")
            print(f"  python scripts/explore_s3_feature_store.py --dataset <folder_name> --peek")
            return 0

        target = data_files[0]["key"]
        if target.endswith(".parquet"):
            read_parquet_head(s3, args.bucket, target, args.rows)
        elif target.endswith(".csv"):
            read_csv_head(s3, args.bucket, target, args.rows)
        else:
            obj = s3.get_object(Bucket=args.bucket, Key=target)
            body = obj["Body"].read(2000)
            print(f"\nFirst bytes of s3://{args.bucket}/{target}:\n{body[:500]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
