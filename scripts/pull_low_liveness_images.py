"""Pull 1 000 ID-document images with the lowest idrnd document-liveness probability.

Records with the sentinel value -999999 (API timeout / error) are excluded.
Images are sourced from S3 via the aws-rekognition cache_key (= S3 object key).

Usage (Colab):
    !python scripts/pull_liveness_images.py \\
        --out-dir /content/low_liveness_images \\
        --limit 1000 \\
        --max-prob 0.5

Output:
    <out-dir>/<image_name>.jpg   – downloaded images
    <out-dir>/manifest.csv       – image path, client_id, doc_prob, s3_key
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import boto3
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
SQL = """
SELECT
    i.client_id,
    (i.response_summary_json->>'aggregate_liveness_probability')::float AS doc_prob,
    r.cache_key AS s3_key
FROM zenka_ke_backend_third_party.idrnd i
JOIN zenka_ke_backend_integration.request r
    ON  r.client_id = i.client_id
    AND r.provider  = 'aws-rekognition-detect-text'
    AND r.cache_key LIKE 'id-doc-front/%%'
WHERE i.provider   = 'idrnd-document-check'
  AND (i.response_summary_json->>'aggregate_liveness_probability') IS NOT NULL
  AND (i.response_summary_json->>'aggregate_liveness_probability')::float <> -999999
  AND (i.response_summary_json->>'aggregate_liveness_probability')::float < %(max_prob)s
ORDER BY doc_prob ASC
LIMIT %(limit)s
"""


def _db_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def _s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-west-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


def pull(out_dir: Path, limit: int, max_prob: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    print(f"Querying DB for up to {limit} records with doc_prob < {max_prob} …")
    conn = _db_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(SQL, {"max_prob": max_prob, "limit": limit})
        rows = cur.fetchall()
    conn.close()
    print(f"  → {len(rows)} rows returned")

    if not rows:
        print("No rows matched. Try raising --max-prob.", file=sys.stderr)
        sys.exit(1)

    bucket = os.environ.get("S3_BUCKET", "sf-zenka-ke-prod-media-svc")
    s3 = _s3_client()

    ok, skip = 0, 0
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["local_path", "client_id", "doc_prob", "s3_key"])
        writer.writeheader()

        for row in rows:
            s3_key: str = row["s3_key"]
            fname = s3_key.replace("/", "_")
            local = out_dir / fname

            if not local.exists():
                try:
                    s3.download_file(bucket, s3_key, str(local))
                    ok += 1
                except Exception as exc:
                    print(f"  SKIP {s3_key}: {exc}")
                    skip += 1
                    continue
                time.sleep(0.02)  # gentle rate-limit
            else:
                ok += 1

            writer.writerow({
                "local_path": str(local),
                "client_id": row["client_id"],
                "doc_prob": row["doc_prob"],
                "s3_key": s3_key,
            })

    print(f"Done. Downloaded {ok} images, skipped {skip}. Manifest: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/low_liveness_images", type=Path)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max-prob", type=float, default=0.5,
                    help="Upper bound on aggregate_liveness_probability (exclusive)")
    args = ap.parse_args()
    pull(args.out_dir, args.limit, args.max_prob)


if __name__ == "__main__":
    main()
