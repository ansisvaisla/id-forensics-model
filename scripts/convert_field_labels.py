"""Convert reviewed OCR field rectangles to canonical cropped-ID coordinates.

Input labels are drawn on full original images. This script uses the reviewed
ID-card corner polygon to warp field rectangles and OCR word boxes into a stable
0-1 canonical ID coordinate system.

Usage:
    python scripts/convert_field_labels.py \
        --label-export data/labels/ocr_field_review_export.json \
        --ocr-csv data/batches/aws_rekognition_detect_text.csv \
        --out data/eval/ocr_field_labels_canonical.csv \
        --templates-out data/eval/ocr_field_templates.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_ocr_audit import _s3_key_from_task  # noqa: E402
from field_localization.geometry import (  # noqa: E402
    Rect,
    homography_full_to_canonical,
    label_studio_rect_to_full,
    transform_rect,
)
from field_localization.providers import OCRWord, from_aws_rekognition_response  # noqa: E402

DEFAULT_LABEL_EXPORT = PROJECT_ROOT / "data" / "labels" / "ocr_field_review_export.json"
DEFAULT_OCR_CSV = PROJECT_ROOT / "data" / "batches" / "aws_rekognition_detect_text.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "eval" / "ocr_field_labels_canonical.csv"
DEFAULT_TEMPLATES_OUT = PROJECT_ROOT / "data" / "eval" / "ocr_field_templates.json"

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
            rows[key] = {"response_summary_json": response}
            if len(rows) == len(wanted_keys):
                break
    return rows


def _all_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for ann in task.get("annotations") or []:
        if ann.get("was_cancelled") or ann.get("skipped"):
            continue
        results.extend(ann.get("result") or [])
    return results


def _choice(task: dict[str, Any], from_name: str) -> str:
    for result in _all_results(task):
        if result.get("from_name") == from_name and result.get("type") == "choices":
            choices = result.get("value", {}).get("choices", [])
            return str(choices[0]) if choices else ""
    return str((task.get("data") or {}).get(f"{from_name}_hint") or "")


def _corners_norm(task: dict[str, Any]) -> list[list[float]] | None:
    for result in _all_results(task):
        if result.get("from_name") == "corners" and result.get("type") == "polygonlabels":
            points = result.get("value", {}).get("points") or []
            if len(points) == 4:
                return [[float(x) / 100.0, float(y) / 100.0] for x, y in points]
    data_corners = (task.get("data") or {}).get("corners_pct") or []
    if len(data_corners) == 4:
        return [[float(x) / 100.0, float(y) / 100.0] for x, y in data_corners]
    return None


def _field_rects(task: dict[str, Any]) -> list[tuple[str, Rect]]:
    rects: list[tuple[str, Rect]] = []
    for result in _all_results(task):
        if result.get("from_name") != "field" or result.get("type") != "rectanglelabels":
            continue
        labels = result.get("value", {}).get("rectanglelabels", [])
        if not labels:
            continue
        label = str(labels[0])
        if label not in FIELD_LABELS:
            continue
        rects.append((label, label_studio_rect_to_full(result.get("value", {}))))
    return rects


def _word_rect(word: OCRWord) -> Rect:
    return Rect(left=word.left, top=word.top, right=word.right, bottom=word.bottom)


def _word_center_inside(word: OCRWord, rect: Rect) -> bool:
    cx = word.left + word.width / 2
    cy = word.top + word.height / 2
    return rect.left <= cx <= rect.right and rect.top <= cy <= rect.bottom


def _words_text(words: list[OCRWord]) -> str:
    ordered = sorted(words, key=lambda word: (round(word.top, 3), word.left))
    return " ".join(word.text for word in ordered).strip()


def _rect_dict(prefix: str, rect: Rect) -> dict[str, float]:
    return {
        f"{prefix}_left": round(rect.left, 6),
        f"{prefix}_top": round(rect.top, 6),
        f"{prefix}_right": round(rect.right, 6),
        f"{prefix}_bottom": round(rect.bottom, 6),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return float(ordered[idx])


def _write_template_summary(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["id_type"], {}).setdefault(row["field_name"], []).append(row)

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for id_type, by_field in grouped.items():
        summary[id_type] = {}
        for field_name, field_rows in by_field.items():
            entry: dict[str, Any] = {"count": len(field_rows)}
            for edge in ("left", "top", "right", "bottom"):
                values = [float(row[f"canonical_{edge}"]) for row in field_rows]
                entry[edge] = {
                    "p10": round(_percentile(values, 0.10), 6),
                    "median": round(statistics.median(values), 6),
                    "p90": round(_percentile(values, 0.90), 6),
                }
            summary[id_type][field_name] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main_with_args(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-export", type=Path, default=DEFAULT_LABEL_EXPORT)
    parser.add_argument("--ocr-csv", type=Path, default=DEFAULT_OCR_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--templates-out", type=Path, default=DEFAULT_TEMPLATES_OUT)
    args = parser.parse_args(argv)

    tasks = _load_tasks(args.label_export)
    wanted = {_s3_key_from_task(task) for task in tasks}
    ocr_rows = _load_ocr_rows(args.ocr_csv, wanted)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for task in tasks:
        s3_key = _s3_key_from_task(task)
        corners = _corners_norm(task)
        field_rects = _field_rects(task)
        ocr_row = ocr_rows.get(s3_key)
        if corners is None or not field_rects or not ocr_row:
            skipped += 1
            continue

        matrix = homography_full_to_canonical(corners)
        words = from_aws_rekognition_response(ocr_row["response_summary_json"])
        id_type = _choice(task, "id_type") or "unknown"
        quality = _choice(task, "quality") or "unknown"

        for field_name, full_rect in field_rects:
            selected_words = [word for word in words if _word_center_inside(word, full_rect)]
            canonical_rect = transform_rect(full_rect, matrix, clip=True)
            canonical_words = [
                {
                    "text": word.text,
                    "confidence": round(word.confidence, 4),
                    **_rect_dict("canonical", transform_rect(_word_rect(word), matrix, clip=True)),
                }
                for word in selected_words
            ]
            rows.append({
                "s3_key": s3_key,
                "id_type": id_type,
                "quality": quality,
                "field_name": field_name,
                "field_text": _words_text(selected_words),
                "word_count": len(selected_words),
                **_rect_dict("full", full_rect),
                **_rect_dict("canonical", canonical_rect),
                "canonical_words_json": json.dumps(canonical_words, ensure_ascii=False),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "s3_key", "id_type", "quality", "field_name", "field_text", "word_count",
        "full_left", "full_top", "full_right", "full_bottom",
        "canonical_left", "canonical_top", "canonical_right", "canonical_bottom",
        "canonical_words_json",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_template_summary(rows, args.templates_out)
    print(f"Canonical field labels written: {args.out} ({len(rows)} fields, {skipped} tasks skipped)")
    print(f"Template summary written: {args.templates_out}")
    return 0


def main() -> int:
    return main_with_args()


if __name__ == "__main__":
    raise SystemExit(main())
