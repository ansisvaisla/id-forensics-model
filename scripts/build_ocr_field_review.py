"""Build a Label Studio import JSON for OCR field rectangle review.

The review UI stays on the original full image because AWS Rekognition OCR boxes
and existing card-corner labels are both in full-image coordinates. Field boxes
are prefilled by projecting the current canonical templates back onto the full
image through the card-corner homography.

Usage:
    python scripts/build_ocr_field_review.py \
        --label-export data/labels/label_studio_ocr_500.json \
        --ocr-csv data/batches/aws_rekognition_detect_text.csv \
        --out data/batches/ocr_field_review_500.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_ocr_audit import _s3_key_from_task  # noqa: E402
from field_localization.geometry import (  # noqa: E402
    Rect,
    homography_canonical_to_full,
    rect_to_label_studio_value,
    transform_rect,
)
from field_localization.providers import from_aws_rekognition_response  # noqa: E402
from field_localization.templates import get_template  # noqa: E402

DEFAULT_LABEL_EXPORT = PROJECT_ROOT / "data" / "labels" / "label_studio_ocr_500.json"
DEFAULT_OCR_CSV = PROJECT_ROOT / "data" / "batches" / "aws_rekognition_detect_text.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "batches" / "ocr_field_review_500.json"

LIVE_QUALITIES = {"good_front", "partial", "blurry"}
FIELD_LABELS = {
    "id_number",
    "serial_number",
    "name",
    "surname",
    "sex",
    "date_of_birth",
    "date_of_issue",
    "place_of_birth",
    "nationality",
    "mrz",
}


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ocr_rows(path: Path, wanted_keys: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row.get("s3_key") or row.get("cache_key") or "").strip()
            if key not in wanted_keys or key in rows:
                continue
            raw = row.get("response_summary_json") or "{}"
            try:
                response = json.loads(raw)
            except json.JSONDecodeError:
                response = json.loads(json.loads(raw))
            rows[key] = {
                "s3_key": key,
                "client_id": row.get("client_id"),
                "application_id": row.get("application_id"),
                "created_at": row.get("created_at"),
                "response_summary_json": response,
            }
            if len(rows) == len(wanted_keys):
                break
    return rows


def _choice(task: dict[str, Any], from_name: str) -> str:
    for result in _annotation_results(task):
        if result.get("from_name") == from_name and result.get("type") == "choices":
            choices = result.get("value", {}).get("choices", [])
            return str(choices[0]) if choices else ""
    return ""


def _corners_pct(task: dict[str, Any]) -> list[list[float]] | None:
    for result in _annotation_results(task):
        if result.get("from_name") == "corners" and result.get("type") == "polygonlabels":
            points = result.get("value", {}).get("points") or []
            if len(points) == 4:
                return [[float(x), float(y)] for x, y in points]
    return None


def _annotation_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for ann in task.get("annotations") or []:
        if ann.get("was_cancelled") or ann.get("skipped"):
            continue
        results.extend(ann.get("result") or [])
    return results


def _copy_prediction_result(result: dict[str, Any]) -> dict[str, Any]:
    copied = {
        "id": str(uuid.uuid4())[:10],
        "from_name": result.get("from_name"),
        "to_name": result.get("to_name", "image"),
        "type": result.get("type"),
        "value": result.get("value", {}),
    }
    for key in ("original_width", "original_height", "image_rotation"):
        if key in result:
            copied[key] = result[key]
    return copied


def _ocr_preview(response: dict[str, Any], max_lines: int = 35) -> str:
    lines = [
        str(item.get("detectedText", "")).strip()
        for item in response.get("textDetections", [])
        if item.get("type") == "LINE" and str(item.get("detectedText", "")).strip()
    ]
    return "\n".join(lines[:max_lines])


def _ocr_word_results(response: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    words = from_aws_rekognition_response(response)
    results: list[dict[str, Any]] = []
    for word in words[:limit]:
        value = rect_to_label_studio_value(
            rect=Rect(left=word.left, top=word.top, right=word.right, bottom=word.bottom),
            label="ocr_word",
            from_name="ocr_word",
        )
        value["id"] = str(uuid.uuid4())[:10]
        results.append(value)
    return results


def _template_field_results(id_type: str, corners_pct: list[list[float]]) -> list[dict[str, Any]]:
    corners_norm = [[x / 100.0, y / 100.0] for x, y in corners_pct]
    matrix = homography_canonical_to_full(corners_norm)
    results: list[dict[str, Any]] = []
    for field_name, zone in get_template(id_type).items():
        if field_name not in FIELD_LABELS:
            continue
        full_rect = transform_rect(zone, matrix, clip=True)
        result = rect_to_label_studio_value(full_rect, field_name, from_name="field")
        result["id"] = str(uuid.uuid4())[:10]
        results.append(result)
    return results


def _build_task(
    task: dict[str, Any],
    ocr_row: dict[str, Any],
    include_ocr_boxes: int,
) -> dict[str, Any] | None:
    quality = _choice(task, "quality")
    id_type = _choice(task, "id_type")
    corners = _corners_pct(task)
    if quality not in LIVE_QUALITIES or not id_type or corners is None:
        return None

    response = ocr_row["response_summary_json"]
    annotation_results = _annotation_results(task)
    copied_results = [
        _copy_prediction_result(result)
        for result in annotation_results
        if result.get("from_name") in {"quality", "id_type", "corners"}
    ]
    copied_results.extend(_template_field_results(id_type, corners))
    copied_results.extend(_ocr_word_results(response, include_ocr_boxes))

    s3_key = _s3_key_from_task(task)
    data = dict(task.get("data") or {})
    data.update({
        "s3_key": s3_key,
        "quality_hint": quality,
        "id_type_hint": id_type,
        "ocr_text": _ocr_preview(response),
        "labeling_notes": (
            "Draw horizontal rectangles around field VALUES only. "
            "Do not include labels like FULL NAMES or DATE OF BIRTH. "
            "For angled text, use a taller horizontal box that covers the value."
        ),
    })

    return {
        "data": data,
        "predictions": [{
            "model_version": "ocr-field-review-v1",
            "score": 0.5,
            "result": copied_results,
        }],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-export", type=Path, default=DEFAULT_LABEL_EXPORT)
    parser.add_argument("--ocr-csv", type=Path, default=DEFAULT_OCR_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--include-ocr-boxes",
        type=int,
        default=0,
        help="Add first N OCR word boxes as visual context. 0 keeps UI cleaner.",
    )
    args = parser.parse_args()

    tasks = _load_tasks(args.label_export)
    wanted = {_s3_key_from_task(task) for task in tasks}
    ocr_rows = _load_ocr_rows(args.ocr_csv, wanted)

    review_tasks: list[dict[str, Any]] = []
    skipped = 0
    for task in tasks:
        key = _s3_key_from_task(task)
        row = ocr_rows.get(key)
        if not row:
            skipped += 1
            continue
        review_task = _build_task(task, row, args.include_ocr_boxes)
        if review_task is None:
            skipped += 1
            continue
        review_tasks.append(review_task)
        if len(review_tasks) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(review_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OCR field review JSON written: {args.out} ({len(review_tasks)} tasks, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
