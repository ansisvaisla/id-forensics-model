"""Filter a mixed Label Studio export down to the OCR audit batch.

Label Studio project exports include every task ever imported. The OCR pilot
batch is the set imported from ``aws_rekognition_detect_text.csv`` with
``batch_label.py --limit 500`` (created on 2026-06-26 in the current project).

Usage:
    python scripts/filter_ocr_label_export.py --report \\
        --input ~/Downloads/project-1-at-2026-06-26-17-16-dafe0e23.json

    python scripts/filter_ocr_label_export.py \\
        --input ~/Downloads/project-1-at-2026-06-26-17-16-dafe0e23.json \\
        --created-on 2026-06-26 \\
        --out data/labels/label_studio_ocr_500.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_ocr_audit import _s3_key_from_task  # noqa: E402

DEFAULT_OCR_CSV = PROJECT_ROOT / "data" / "batches" / "aws_rekognition_detect_text.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "labels" / "label_studio_ocr_500.json"
OCR_BATCH_CREATED_ON = "2026-06-26"


def _load_ocr_keys(csv_path: Path) -> set[str]:
    keys: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row.get("s3_key") or row.get("cache_key") or "").strip()
            if key:
                keys.add(key)
    return keys


def _task_profile(task: dict[str, Any]) -> tuple[str, ...]:
    names: set[str] = set()
    for ann in task.get("annotations") or []:
        if ann.get("was_cancelled"):
            continue
        for result in ann.get("result") or []:
            name = result.get("from_name")
            if name:
                names.add(name)
    return tuple(sorted(names))


def _created_on(task: dict[str, Any]) -> str:
    return (task.get("created_at") or "")[:10]


def _ocr_keys_present(tasks: list[dict[str, Any]], ocr_keys: set[str]) -> int:
    return sum(1 for task in tasks if _s3_key_from_task(task) in ocr_keys)


def _summarize_export(tasks: list[dict[str, Any]]) -> None:
    by_day: dict[str, int] = {}
    for task in tasks:
        day = _created_on(task)
        by_day[day] = by_day.get(day, 0) + 1

    print(f"Label Studio export tasks: {len(tasks)}")
    print("Tasks by created_at:")
    for day in sorted(by_day):
        print(f"  {day}: {by_day[day]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--created-on",
        default=OCR_BATCH_CREATED_ON,
        help=f"Keep only tasks created on this date (default: {OCR_BATCH_CREATED_ON})",
    )
    parser.add_argument("--ocr-csv", type=Path, default=DEFAULT_OCR_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", action="store_true", help="Print stats only")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    _summarize_export(tasks)

    matched = [task for task in tasks if _created_on(task) == args.created_on]
    print(f"\nFiltered to created_on={args.created_on}: {len(matched)} tasks")

    profiles: dict[tuple[str, ...], int] = {}
    for task in matched:
        profile = _task_profile(task)
        profiles[profile] = profiles.get(profile, 0) + 1
    print("Label profiles:")
    for profile, count in sorted(profiles.items(), key=lambda item: -item[1]):
        print(f"  {profile}: {count}")

    if args.ocr_csv.is_file():
        ocr_keys = _load_ocr_keys(args.ocr_csv)
        overlap = _ocr_keys_present(matched, ocr_keys)
        print(f"Tasks with OCR row in {args.ocr_csv.name}: {overlap}/{len(matched)}")
    else:
        print(f"OCR CSV not found, skipping overlap check: {args.ocr_csv}")

    if args.report:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFiltered export written: {args.out} ({len(matched)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
