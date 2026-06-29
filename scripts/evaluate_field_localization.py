"""Evaluate Stage 4 field localization against a reviewed CSV.

Expected input is an OCR audit CSV with optional expected_* columns added by a
human reviewer, for example:
    expected_id_number, expected_name, expected_date_of_birth

Usage:
    python scripts/evaluate_field_localization.py --input data/eval/ocr_audit_reviewed.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = (
    "id_number",
    "serial_number",
    "name",
    "surname",
    "sex",
    "date_of_birth",
    "date_of_issue",
    "place_of_birth",
    "nationality",
)


def _norm(value: str | None) -> str:
    return " ".join((value or "").upper().replace(".", "/").replace("-", "/").split())


def evaluate(input_path: Path) -> dict:
    rows = list(csv.DictReader(input_path.open(newline="", encoding="utf-8")))
    by_field: dict[str, Counter] = {field: Counter() for field in FIELDS}
    by_type: dict[str, Counter] = {}

    for row in rows:
        id_type = row.get("id_type", "unknown") or "unknown"
        by_type.setdefault(id_type, Counter())
        for field in FIELDS:
            expected = _norm(row.get(f"expected_{field}"))
            if not expected:
                continue
            pred = _norm(row.get(field))
            ok = pred == expected
            by_field[field]["total"] += 1
            by_field[field]["correct"] += int(ok)
            by_type[id_type]["total"] += 1
            by_type[id_type]["correct"] += int(ok)

    field_metrics = {}
    for field, counts in by_field.items():
        total = counts["total"]
        correct = counts["correct"]
        field_metrics[field] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else None,
        }

    type_metrics = {}
    for id_type, counts in by_type.items():
        total = counts["total"]
        correct = counts["correct"]
        type_metrics[id_type] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else None,
        }

    return {"rows": len(rows), "fields": field_metrics, "id_types": type_metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate(args.input)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
