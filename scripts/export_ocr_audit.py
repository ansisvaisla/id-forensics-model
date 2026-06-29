"""Export a Stage 4 OCR field-localization audit CSV from stored AWS Rekognition logs.

This script does not call OCR. It uses existing rows in
zenka_ke_backend_integration.request where provider = 'aws-rekognition-detect-text'
and joins them to Label Studio tasks by the image cache_key / S3 object key.

Usage:
    python scripts/export_ocr_audit.py --limit 500
    python scripts/export_ocr_audit.py --id-type legacy --out data/eval/ocr_audit_legacy.csv
    python scripts/export_ocr_audit.py --ocr-csv data/batches/aws_rekognition_detect_text.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from db_zenka_ke import _connect  # noqa: E402
from orchestration.results import ExtractedFields  # noqa: E402

from field_localization import run as run_field_localization  # noqa: E402
from field_localization.providers import from_aws_rekognition_response  # noqa: E402

_REQUIRED_EXTRACTED_FIELDS = ("date_of_issue", "place_of_birth", "serial_number")

DEFAULT_EXPORT = PROJECT_ROOT / "data" / "labels" / "label_studio_export.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "eval" / "ocr_audit.csv"

OCR_QUERY = """
SELECT cache_key, response_summary_json, created_at
FROM zenka_ke_backend_integration.request
WHERE provider = 'aws-rekognition-detect-text'
  AND cache_key = ANY(%(keys)s)
ORDER BY created_at DESC
"""


def _s3_key_from_task(task: dict[str, Any]) -> str:
    image_uri = task.get("data", {}).get("image", "") or ""
    if image_uri.startswith("s3://"):
        without_prefix = image_uri.removeprefix("s3://")
        _, _, key = without_prefix.partition("/")
        return key
    if image_uri.startswith("http"):
        from urllib.parse import urlparse
        key = urlparse(image_uri).path.lstrip("/")
        if key.startswith("sf-zenka"):
            key = key.split("/", 1)[-1]
        return key
    file_upload = task.get("file_upload", "") or ""
    flat = file_upload.split("-", 1)[1] if "-" in file_upload else file_upload
    stem, ext = flat.rsplit(".", 1) if "." in flat else (flat, "jpg")
    parts = stem.split("_")
    if len(parts) >= 4 and len(parts[0]) == 4 and parts[0].isdigit():
        return f"id-doc-front/{parts[0]}/{parts[1]}/{parts[2]}/{'_'.join(parts[3:])}.{ext}"
    return flat


def _choice(task: dict[str, Any], from_name: str) -> str:
    for ann in task.get("annotations", []):
        if ann.get("was_cancelled") or ann.get("skipped"):
            continue
        for result in ann.get("result", []):
            if result.get("from_name") == from_name and result.get("type") == "choices":
                choices = result.get("value", {}).get("choices", [])
                return str(choices[0]) if choices else ""
    return ""


def _load_tasks(export_path: Path, id_type_filter: str | None, limit: int) -> list[dict[str, str]]:
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for task in tasks:
        id_type = _choice(task, "id_type")
        quality = _choice(task, "quality")
        if id_type_filter and id_type != id_type_filter:
            continue
        if quality not in {"good_front", "partial", "blurry"}:
            continue
        key = _s3_key_from_task(task)
        if not key.startswith("id-doc-front/"):
            continue
        rows.append({"s3_key": key, "id_type": id_type or "unknown", "quality": quality})
        if len(rows) >= limit:
            break
    return rows


def _fetch_ocr(keys: list[str]) -> dict[str, dict[str, Any]]:
    if not keys:
        return {}
    with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(OCR_QUERY, {"keys": keys})
        rows = [dict(row) for row in cur.fetchall()]
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest.setdefault(row["cache_key"], row)
    return latest


def _load_ocr_csv(csv_path: Path, keys: list[str]) -> dict[str, dict[str, Any]]:
    """Stream a manually exported OCR CSV and keep only requested S3 keys."""
    wanted = set(keys)
    if not wanted:
        return {}

    latest: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"s3_key", "response_summary_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OCR CSV missing required columns: {sorted(missing)}")

        for row in reader:
            s3_key = (row.get("s3_key") or "").strip()
            if s3_key not in wanted or s3_key in latest:
                continue

            raw_json = row.get("response_summary_json") or "{}"
            try:
                response_summary_json = json.loads(raw_json)
            except json.JSONDecodeError:
                # Some DBeaver exports double-quote JSON strings. Try one more decode.
                response_summary_json = json.loads(json.loads(raw_json))

            latest[s3_key] = {
                "cache_key": s3_key,
                "client_id": row.get("client_id"),
                "application_id": row.get("application_id"),
                "created_at": row.get("created_at"),
                "response_summary_json": response_summary_json,
            }

            if len(latest) == len(wanted):
                break

    return latest


def _verify_extracted_fields_schema() -> None:
    missing = [
        name for name in _REQUIRED_EXTRACTED_FIELDS
        if name not in ExtractedFields.__dataclass_fields__
    ]
    if missing:
        raise RuntimeError(
            "Stale orchestration.results.ExtractedFields on this runtime. "
            f"Missing: {missing}. Run `git pull` in Colab section 1/8, then "
            "Runtime → Restart session and re-run section 8."
        )


def _field_value(fields: ExtractedFields, name: str) -> str | None:
    return getattr(fields, name, None)


def main() -> int:
    _verify_extracted_fields_schema()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--id-type", default=None)
    parser.add_argument(
        "--ocr-csv",
        type=Path,
        default=None,
        help="Manual DB export with s3_key,response_summary_json columns. "
             "If omitted, the script queries Postgres directly.",
    )
    args = parser.parse_args()

    if not args.label_export.is_file():
        print(f"Label export not found: {args.label_export}", file=sys.stderr)
        return 1

    tasks = _load_tasks(args.label_export, args.id_type, args.limit)
    keys = [row["s3_key"] for row in tasks]
    if args.ocr_csv:
        if not args.ocr_csv.is_file():
            print(f"OCR CSV not found: {args.ocr_csv}", file=sys.stderr)
            return 1
        ocr_by_key = _load_ocr_csv(args.ocr_csv, keys)
    else:
        ocr_by_key = _fetch_ocr(keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dummy_image = np.zeros((1, 1, 3), dtype=np.uint8)
    fieldnames = [
        "s3_key", "id_type", "quality", "ocr_found", "ocr_word_count",
        "label", "confidence", "id_number", "serial_number", "name", "surname",
        "sex", "date_of_birth", "date_of_issue", "place_of_birth", "nationality",
    ]
    found_count = 0
    parsed_count = 0
    error_count = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            ocr_row = ocr_by_key.get(task["s3_key"])
            if not ocr_row:
                writer.writerow({**task, "ocr_found": False, "ocr_word_count": 0})
                continue
            found_count += 1
            try:
                words = from_aws_rekognition_response(ocr_row["response_summary_json"])
                result = run_field_localization(
                    dummy_image, id_type=task["id_type"], ocr_words=words,
                )
                fields = result.extracted_fields
                writer.writerow({
                    **task,
                    "ocr_found": True,
                    "ocr_word_count": len(words),
                    "label": result.label,
                    "confidence": round(result.field_extraction_confidence, 4),
                    "id_number": _field_value(fields, "id_number"),
                    "serial_number": _field_value(fields, "serial_number"),
                    "name": _field_value(fields, "name"),
                    "surname": _field_value(fields, "surname"),
                    "sex": _field_value(fields, "sex"),
                    "date_of_birth": _field_value(fields, "date_of_birth"),
                    "date_of_issue": _field_value(fields, "date_of_issue"),
                    "place_of_birth": _field_value(fields, "place_of_birth"),
                    "nationality": _field_value(fields, "nationality"),
                })
                parsed_count += 1
            except Exception as exc:
                error_count += 1
                print(f"PARSE ERROR {task['s3_key']}: {exc}", file=sys.stderr)
                writer.writerow({
                    **task,
                    "ocr_found": True,
                    "ocr_word_count": 0,
                    "label": "failed",
                    "confidence": 0.0,
                })

    print(
        f"OCR audit written: {args.out} "
        f"({len(tasks)} candidate rows, {found_count} with OCR, "
        f"{parsed_count} parsed, {error_count} errors)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
