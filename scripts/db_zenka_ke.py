"""PostgreSQL queries for Zenka Kenya ID images (postgres2)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ATTACHMENT_SCHEMA = "zenka_ke_backend_crm"
INTEGRATION_SCHEMA = "zenka_ke_backend_integration"

SUB_TYPE_ID_FRONT = "ID_DOC_FRONT"
SUB_TYPE_SELFIE = "SELFIE"

REQUEST_QUERY = """
SELECT
    (params_json->>'attachmentId')::bigint AS attachment_id,
    params_json->>'objectKey' AS s3_key,
    COALESCE(params_json->>'bucketName', %(default_bucket)s) AS bucket,
    created_at AS request_created_at
FROM {integration_schema}.request
WHERE provider = %(provider)s
  AND created_at >= NOW() - make_interval(hours => %(hours)s)
  AND params_json->>'objectKey' LIKE %(key_prefix)s
ORDER BY created_at DESC
LIMIT %(fetch_limit)s
"""

ATTACHMENT_QUERY = """
SELECT
    id AS attachment_id,
    client_id,
    sub_type,
    file_name,
    cloud_file_id,
    created_at AS attachment_created_at
FROM {attachment_schema}.attachment
WHERE id = ANY(%(attachment_ids)s)
  AND deleted = false
"""


@dataclass(frozen=True)
class ZenkaAttachmentRow:
    attachment_id: int
    client_id: int
    sub_type: str
    s3_key: str
    bucket: str
    file_name: str | None
    attachment_created_at: datetime | None
    request_created_at: datetime


def _database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("ZENKA_KE_DATABASE_URL") or os.getenv("DATABASE_URL")
    if url:
        return url

    host = os.getenv("ZENKA_KE_DB_HOST", "").strip()
    user = os.getenv("ZENKA_KE_DB_USER", "").strip()
    password = os.getenv("ZENKA_KE_DB_PASSWORD", "")
    name = os.getenv("ZENKA_KE_DB_NAME", "").strip()
    port = os.getenv("ZENKA_KE_DB_PORT", "5432").strip()

    if host and user and password and name:
        return (
            f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{name}"
        )

    raise SystemExit(
        "Set Zenka Kenya DB credentials in .env:\n"
        "  either ZENKA_KE_DB_HOST, ZENKA_KE_DB_USER, ZENKA_KE_DB_PASSWORD, "
        "ZENKA_KE_DB_NAME\n"
        "  or a full ZENKA_KE_DATABASE_URL"
    )


def _connect():
    return psycopg2.connect(_database_url())


def _provider_for_sub_type(sub_type: str) -> tuple[str, str]:
    if sub_type == SUB_TYPE_ID_FRONT:
        return "aws-rekognition-detect-text", "id-doc-front/%"
    if sub_type == SUB_TYPE_SELFIE:
        return "aws-rekognition-index-faces", "selfie/%"
    raise ValueError(f"Unsupported sub_type: {sub_type}")


def _dedupe_requests(rows: list[dict]) -> list[dict]:
    seen: set[int] = set()
    deduped: list[dict] = []
    for row in rows:
        attachment_id = row["attachment_id"]
        if attachment_id in seen:
            continue
        seen.add(attachment_id)
        deduped.append(row)
    return deduped


def fetch_attachments_with_s3_keys(
    sub_type: str,
    *,
    limit: int = 10,
    hours: int = 24,
    default_bucket: str = "sf-zenka-ke-prod-media-svc",
) -> list[ZenkaAttachmentRow]:
    """Fetch recent attachments with S3 keys via integration.request."""
    provider, key_prefix = _provider_for_sub_type(sub_type)
    fetch_limit = max(limit * 3, limit)

    with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            REQUEST_QUERY.format(integration_schema=INTEGRATION_SCHEMA),
            {
                "provider": provider,
                "hours": hours,
                "key_prefix": key_prefix,
                "fetch_limit": fetch_limit,
                "default_bucket": default_bucket,
            },
        )
        request_rows = _dedupe_requests([dict(row) for row in cur.fetchall()])[:limit]
        if not request_rows:
            return []

        attachment_ids = [row["attachment_id"] for row in request_rows]
        cur.execute(
            ATTACHMENT_QUERY.format(attachment_schema=ATTACHMENT_SCHEMA),
            {"attachment_ids": attachment_ids},
        )
        attachments = {row["attachment_id"]: dict(row) for row in cur.fetchall()}

    results: list[ZenkaAttachmentRow] = []
    for request_row in request_rows:
        attachment = attachments.get(request_row["attachment_id"])
        if attachment is None or attachment["sub_type"] != sub_type:
            continue
        results.append(
            ZenkaAttachmentRow(
                attachment_id=request_row["attachment_id"],
                client_id=attachment["client_id"],
                sub_type=attachment["sub_type"],
                s3_key=request_row["s3_key"],
                bucket=request_row["bucket"],
                file_name=attachment.get("file_name"),
                attachment_created_at=attachment.get("attachment_created_at"),
                request_created_at=request_row["request_created_at"],
            )
        )
    return results
