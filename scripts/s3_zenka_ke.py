"""S3 helpers for Zenka Kenya ID images in sf-zenka-ke-prod-media-svc."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZENKA_KE_BUCKET = "sf-zenka-ke-prod-media-svc"
ZENKA_KE_PREFIX_ID_FRONT = "id-doc-front"
ZENKA_KE_PREFIX_SELFIE = "selfie"

ImageKind = Literal["id_front", "selfie"]


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def s3_client():
    _load_env()
    return boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))


def bucket_name() -> str:
    _load_env()
    return os.getenv("AWS_S3_BUCKET", ZENKA_KE_BUCKET)


def resolve_s3_key(file_path: str, kind: ImageKind | None = None) -> str:
    """Build the full S3 object key from a DB `file` value or partial path."""
    key = file_path.lstrip("/")
    if key.startswith(f"{ZENKA_KE_PREFIX_ID_FRONT}/") or key.startswith(f"{ZENKA_KE_PREFIX_SELFIE}/"):
        return key

    if kind == "id_front":
        return f"{ZENKA_KE_PREFIX_ID_FRONT}/{key}"
    if kind == "selfie":
        return f"{ZENKA_KE_PREFIX_SELFIE}/{key}"

    lowered = key.lower()
    if "selfie" in lowered:
        return f"{ZENKA_KE_PREFIX_SELFIE}/{key}"
    if "front" in lowered or "id-doc" in lowered:
        return f"{ZENKA_KE_PREFIX_ID_FRONT}/{key}"

    return key


def candidate_keys(file_path: str, kind: ImageKind | None = None) -> list[str]:
    """Return likely S3 keys to probe, most specific first."""
    primary = resolve_s3_key(file_path, kind=kind)
    candidates = [primary]
    if kind is None:
        for prefix in (ZENKA_KE_PREFIX_ID_FRONT, ZENKA_KE_PREFIX_SELFIE):
            alt = f"{prefix}/{file_path.lstrip('/')}"
            if alt not in candidates:
                candidates.append(alt)
    return candidates


def download_image(
    file_path: str,
    dest: Path,
    *,
    kind: ImageKind | None = None,
    bucket: str | None = None,
) -> str:
    """Download an image; returns the S3 key that succeeded."""
    s3 = s3_client()
    bucket = bucket or bucket_name()
    last_error: ClientError | None = None

    for key in candidate_keys(file_path, kind=kind):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            return key
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No object found for {file_path!r} in s3://{bucket}/")


def download_by_key(s3_key: str, dest: Path, *, bucket: str | None = None) -> str:
    """Download using a full S3 object key from integration.request params_json."""
    s3 = s3_client()
    bucket = bucket or bucket_name()
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(dest))
    return s3_key
