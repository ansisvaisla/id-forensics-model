"""Reservoir-sample S3 paths from a large manifest CSV without loading it all into RAM."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "manifests" / "zenka_ke_id_front.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "pilot_500.csv"


def _extract_s3_key(row: dict[str, str]) -> str | None:
    for col in ("s3_key", "cache_key", "path"):
        value = row.get(col, "").strip()
        if value:
            return value
    if row:
        return next(iter(row.values())).strip() or None
    return None


def reservoir_sample(
    input_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> int:
    rng = random.Random(seed)
    reservoir: list[str] = []
    total_rows = 0

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _extract_s3_key(row)
            if not key:
                continue
            total_rows += 1
            if len(reservoir) < sample_size:
                reservoir.append(key)
            else:
                pick = rng.randint(0, total_rows - 1)
                if pick < sample_size:
                    reservoir[pick] = key

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["s3_key"])
        for key in reservoir:
            writer.writerow([key])

    return total_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Reservoir-sample paths from a large manifest CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=500, help="Number of paths to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    total = reservoir_sample(args.input, args.output, args.size, args.seed)
    print(f"Sampled {args.size} paths from {total:,} rows")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
