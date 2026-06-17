"""Fetch ID document rows from ERP PostgreSQL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERY = """
SELECT
    cid.id,
    cid.client_id,
    cid.type AS doc_type,
    cid.file AS s3_key,
    cid.source,
    cid.document_type,
    cid.document_number,
    cid.createdat
FROM {schema}.client_id_document cid
WHERE cid.deletedat IS NULL
  AND cid.file IS NOT NULL
ORDER BY cid.createdat DESC
LIMIT %(limit)s
"""

AWS_OCR_QUERY = """
SELECT
    cid.id AS document_id,
    cid.client_id,
    cid.file AS s3_key,
    aipr.type AS aws_operation,
    aipr.document_type,
    aipr.application_id,
    aipr.createdat AS aws_processed_at,
    aipr.raw_response
FROM {schema}.client_id_document cid
JOIN {schema}.aws_image_process_result aipr
    ON aipr.document_id = cid.id
WHERE cid.id = %(document_id)s
  AND aipr.not_finished = false
ORDER BY aipr.createdat DESC
"""


def _connect():
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit(
            "Set DATABASE_URL in .env, e.g.\n"
            "DATABASE_URL=postgresql://user:pass@host:5432/dbname"
        )
    return psycopg2.connect(url)


def fetch_documents(schema: str, limit: int) -> list[dict]:
    with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(DEFAULT_QUERY.format(schema=schema), {"limit": limit})
        return [dict(row) for row in cur.fetchall()]


def fetch_aws_results(schema: str, document_id: int) -> list[dict]:
    with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(AWS_OCR_QUERY.format(schema=schema), {"document_id": document_id})
        rows = [dict(row) for row in cur.fetchall()]
        for row in rows:
            if row.get("raw_response") is not None:
                row["raw_response"] = row["raw_response"]
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Query client_id_document rows from ERP DB")
    parser.add_argument("--schema", default=os.getenv("ERP_SCHEMA", "erp_finbro_ph"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--document-id", type=int, help="Also fetch AWS OCR rows for this id")
    parser.add_argument("--download", action="store_true", help="Download s3_key to ./data/")
    args = parser.parse_args()

    rows = fetch_documents(args.schema, args.limit)
    print(json.dumps(rows, default=str, indent=2))

    if args.document_id:
        aws_rows = fetch_aws_results(args.schema, args.document_id)
        print("\n--- AWS OCR results ---")
        print(json.dumps(aws_rows, default=str, indent=2))

    if args.download:
        import boto3

        load_dotenv(PROJECT_ROOT / ".env")
        bucket = os.getenv("AWS_S3_BUCKET")
        if not bucket:
            print("Set AWS_S3_BUCKET in .env to download files.", file=sys.stderr)
            return 1

        out_dir = PROJECT_ROOT / "data"
        out_dir.mkdir(exist_ok=True)
        s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))

        for row in rows:
            key = row["s3_key"]
            dest = out_dir / Path(key).name
            s3.download_file(bucket, key, str(dest))
            print(f"Downloaded s3://{bucket}/{key} -> {dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
