"""Verify AWS credentials and optional S3 bucket access."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _session() -> boto3.Session:
    load_dotenv(PROJECT_ROOT / ".env")
    profile = os.getenv("AWS_PROFILE")
    if profile:
        return boto3.Session(profile_name=profile, region_name=os.getenv("AWS_DEFAULT_REGION"))
    return boto3.Session(region_name=os.getenv("AWS_DEFAULT_REGION"))


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    region = os.getenv("AWS_DEFAULT_REGION", "not set")
    bucket = os.getenv("AWS_S3_BUCKET")

    print("=== AWS access check ===")
    print(f"Region: {region}")
    print(f"S3 bucket (from .env): {bucket or 'not set'}")

    has_key = bool(os.getenv("AWS_ACCESS_KEY_ID"))
    has_token = bool(os.getenv("AWS_SESSION_TOKEN"))
    has_profile = bool(os.getenv("AWS_PROFILE"))

    if has_key and not has_token:
        print(
            "\nWARNING: AWS_ACCESS_KEY_ID is set but AWS_SESSION_TOKEN is missing.\n"
            "IAM Identity Center credentials require all three values from the modal."
        )

    try:
        session = _session()
        sts = session.client("sts")
        identity = sts.get_caller_identity()
    except ProfileNotFound as exc:
        print(f"\nFAIL: AWS profile not found: {exc}")
        return 1
    except NoCredentialsError:
        print("\nFAIL: No credentials found. Fill in .env (see .env.example).")
        return 1
    except ClientError as exc:
        print(f"\nFAIL: STS error: {exc}")
        return 1

    print("\nOK: Authenticated")
    print(f"  Account: {identity['Account']}")
    print(f"  ARN:     {identity['Arn']}")

    if not bucket:
        print("\nSkip S3 test: set AWS_S3_BUCKET in .env when you know the bucket name.")
        return 0

    s3 = session.client("s3")
    prefix = os.getenv("AWS_S3_PREFIX", "")
    list_prefix = f"{bucket}/{prefix}" if prefix else bucket
    print(f"\nTesting S3 list: s3://{list_prefix}")

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        if keys:
            print("OK: Listed objects (up to 5):")
            for key in keys:
                print(f"  - {key}")
        else:
            print("OK: List succeeded but no objects under that prefix.")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        print(f"FAIL: S3 list error ({code}): {exc}")
        print("  Common fix: ask IT for s3:ListBucket and s3:GetObject on this bucket.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
