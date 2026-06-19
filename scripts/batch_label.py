"""Batch labeling loop — generate a Label Studio import JSON with pipeline pre-annotations.

Workflow
────────
Option A — DB mode (default, requires network access to postgres):
  python scripts/batch_label.py --limit 1000 --hours 720

Option B — CSV mode (no DB needed — export candidates.csv from DBeaver first):
  python scripts/batch_label.py --from-csv data/batches/candidates.csv --limit 1000

Skip inference (no predictions, fastest):
  python scripts/batch_label.py --from-csv data/batches/candidates.csv --skip-inference

Performance
───────────
- Images downloaded in parallel (default 32 threads) → ~20-30s for 1000 images
- Presigned URLs generated in parallel
- Pipeline inference sequential on GPU → ~0.1s/img on T4 = ~2min for 1000 images
- Textract (field_extractor) skipped automatically — set SKIP_FIELD_EXTRACTOR=1 or
  use --skip-field-extractor flag (no AWS Textract permission needed)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_FILE = PROJECT_ROOT / "data" / "labels" / "label_studio_export.json"
BATCHES_DIR = PROJECT_ROOT / "data" / "batches"
_DRIVE_BATCHES_DIR = Path("/content/drive/MyDrive/id-forensics/data/batches")
_MODEL_VERSION = "id-forensics-pipeline"


# ── Colab secret injection ────────────────────────────────────────────────────

def _inject_colab_db_secrets() -> None:
    try:
        from google.colab import userdata  # type: ignore[import-untyped]
    except ImportError:
        return

    def _get(name: str) -> str:
        try:
            return userdata.get(name) or ""
        except Exception:
            return ""

    host = _get("ZENKA_KE_DB_HOST").strip()
    user = _get("ZENKA_KE_DB_USER").strip()
    pwd = _get("ZENKA_KE_DB_PASSWORD").strip()
    name = _get("ZENKA_KE_DB_NAME").strip()
    port = _get("ZENKA_KE_DB_PORT").strip() or "5432"

    if host and user and pwd and name:
        from urllib.parse import quote
        dsn = f"postgresql://{quote(user, safe='')}:{quote(pwd, safe='')}@{host}:{port}/{name}"
        os.environ["ZENKA_KE_DATABASE_URL"] = dsn
    else:
        url = _get("ZENKA_KE_DATABASE_URL")
        if url:
            os.environ["ZENKA_KE_DATABASE_URL"] = url


def _default_batches_dir() -> Path:
    if _DRIVE_BATCHES_DIR.parent.parent.is_dir():
        return _DRIVE_BATCHES_DIR
    return BATCHES_DIR


# ── Already-labeled helpers ───────────────────────────────────────────────────

def _flat_name(file_upload: str) -> str:
    parts = file_upload.split("-", 1)
    return parts[1] if len(parts) == 2 else file_upload


def _already_labeled(export_path: Path) -> set[str]:
    if not export_path.is_file():
        return set()
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    stems: set[str] = set()
    for task in tasks:
        fu = task.get("file_upload", "")
        if fu:
            stems.add(Path(_flat_name(fu)).stem)
    return stems


def _flat_from_s3_key(s3_key: str) -> str:
    rel = s3_key.removeprefix("id-doc-front/").lstrip("/")
    return rel.replace("/", "_")


def _stem_from_s3_key(s3_key: str) -> str:
    return Path(_flat_from_s3_key(s3_key)).stem


# ── Candidate fetch ───────────────────────────────────────────────────────────

def _fetch_candidates_db(hours: int, limit: int, labeled_stems: set[str]) -> list:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from db_zenka_ke import fetch_attachments_with_s3_keys, SUB_TYPE_ID_FRONT
    rows = fetch_attachments_with_s3_keys(SUB_TYPE_ID_FRONT, limit=limit * 3, hours=hours)
    return [r for r in rows if _stem_from_s3_key(r.s3_key) not in labeled_stems][:limit]


def _fetch_candidates_csv(csv_path: Path, limit: int, labeled_stems: set[str]) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s3_key = row.get("s3_key", "").strip()
            if not s3_key or _stem_from_s3_key(s3_key) in labeled_stems:
                continue
            rows.append({
                "s3_key": s3_key,
                "bucket": row.get("bucket", "sf-zenka-ke-prod-media-svc").strip(),
            })
            if len(rows) >= limit:
                break
    return rows


# ── Parallel S3 operations ────────────────────────────────────────────────────

def _s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", config=Config(
        signature_version="s3v4",
        max_pool_connections=64,
    ))


def _download_all(candidates: list, workers: int = 32) -> dict[int, Optional[bytes]]:
    """Download all images in parallel. Returns {index: bytes_or_None}."""
    client = _s3_client()

    def _one(idx_key_bucket: tuple) -> tuple[int, Optional[bytes]]:
        idx, s3_key, bucket = idx_key_bucket
        try:
            buf = io.BytesIO()
            client.download_fileobj(bucket, s3_key, buf)
            return idx, buf.getvalue()
        except Exception as exc:
            print(f"  [download] FAIL {s3_key}: {exc}", file=sys.stderr)
            return idx, None

    items = [
        (i, (row.s3_key if hasattr(row, "s3_key") else row["s3_key"]),
         (row.bucket if hasattr(row, "bucket") else row["bucket"]))
        for i, row in enumerate(candidates)
    ]

    results: dict[int, Optional[bytes]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, item): item[0] for item in items}
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="img")
        except ImportError:
            pbar = as_completed(futures)  # type: ignore
        for future in pbar:
            idx, data = future.result()
            results[idx] = data
    return results


def _generate_urls_all(candidates: list, expiry: int, workers: int = 32) -> dict[int, str]:
    """Generate presigned URLs in parallel. Returns {index: url}."""
    client = _s3_client()

    def _one(idx_key_bucket: tuple) -> tuple[int, str]:
        idx, s3_key, bucket = idx_key_bucket
        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expiry,
            )
            return idx, url
        except Exception as exc:
            print(f"  [url] FAIL {s3_key}: {exc}", file=sys.stderr)
            return idx, ""

    items = [
        (i, (row.s3_key if hasattr(row, "s3_key") else row["s3_key"]),
         (row.bucket if hasattr(row, "bucket") else row["bucket"]))
        for i, row in enumerate(candidates)
    ]

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, item): item[0] for item in items}
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Presigning URLs", unit="img")
        except ImportError:
            pbar = as_completed(futures)  # type: ignore
        for future in pbar:
            idx, url = future.result()
            results[idx] = url
    return results


# ── Pipeline inference ────────────────────────────────────────────────────────

def _run_pipeline(image_bytes: bytes) -> Optional[object]:
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import orchestration
        return orchestration.run(image_bytes)
    except Exception as exc:
        print(f"  [pipeline] FAIL: {exc}", file=sys.stderr)
        return None


# ── LS prediction builder ─────────────────────────────────────────────────────

def _quality_from_result(result) -> str:
    if result.crop and result.crop.label == "selfie_instead_of_document":
        return "selfie"
    if result.is_screen_replay:
        return "screen"
    if result.is_printout:
        return "printout"
    if result.crop and result.crop.label in ("no_id_detected", "invalid_crop"):
        return "garbage"
    if result.is_partial_document:
        return "partial"
    return "good_front"


def _to_ls_predictions(result) -> list[dict]:
    quality = _quality_from_result(result)
    id_type_label = "unknown_id"
    if result.id_type is not None:
        raw = result.id_type.id_type
        id_type_label = raw if raw != "unknown" else "unknown_id"

    if result.is_screen_replay and result.presentation_attack:
        confidence = result.presentation_attack.screen_score
    elif result.is_printout and result.presentation_attack:
        confidence = result.presentation_attack.printout_score
    elif result.id_type is not None:
        confidence = result.id_type.confidence
    else:
        confidence = 0.5

    return [{
        "model_version": _MODEL_VERSION,
        "score": round(float(confidence), 4),
        "result": [
            {"from_name": "quality", "to_name": "image", "type": "choices",
             "value": {"choices": [quality]}},
            {"from_name": "id_type", "to_name": "image", "type": "choices",
             "value": {"choices": [id_type_label]}},
        ],
    }]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    _inject_colab_db_secrets()
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Generate Label Studio import JSON with pipeline pre-annotations"
    )
    parser.add_argument("--from-csv", type=Path, default=None, metavar="CSV",
                        help="Read candidates from CSV instead of DB")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--hours", type=int, default=720,
                        help="DB look-back window in hours (default 720 = 30 days)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip pipeline — no predictions, much faster")
    parser.add_argument("--skip-field-extractor", action="store_true",
                        help="Skip AWS Textract stage (set automatically if no Textract access)")
    parser.add_argument("--url-expiry", type=int, default=604_800,
                        help="Presigned URL lifetime in seconds (default 7 days)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel download/URL threads (default 32)")
    parser.add_argument("--export", type=Path, default=EXPORT_FILE)
    args = parser.parse_args()

    # Skip Textract if flagged or env var already set
    if args.skip_field_extractor:
        os.environ["SKIP_FIELD_EXTRACTOR"] = "1"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (_default_batches_dir() / f"{ts}_batch.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: already-labeled ───────────────────────────────────────────────
    labeled_stems = _already_labeled(args.export)
    print(f"Already labeled: {len(labeled_stems)} images")

    # ── Step 2: fetch candidates ──────────────────────────────────────────────
    if args.from_csv:
        print(f"Reading candidates from CSV: {args.from_csv}")
        try:
            candidates = _fetch_candidates_csv(args.from_csv, args.limit, labeled_stems)
        except Exception as exc:
            print(f"CSV read failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"Querying DB (last {args.hours}h)...")
        try:
            candidates = _fetch_candidates_db(args.hours, args.limit, labeled_stems)
        except SystemExit as exc:
            print(f"DB connection failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"DB query failed: {exc}", file=sys.stderr)
            return 1

    if not candidates:
        print("No new unlabeled candidates found.")
        return 0

    print(f"Candidates to label: {len(candidates)}")

    # ── Step 3: parallel download + URL generation ────────────────────────────
    print(f"Downloading {len(candidates)} images in parallel ({args.workers} threads)...")
    image_data = _download_all(candidates, workers=args.workers) if not args.skip_inference else {}

    print("Generating presigned URLs in parallel...")
    urls = _generate_urls_all(candidates, args.url_expiry, workers=args.workers)

    # ── Step 4: pipeline inference + build tasks ──────────────────────────────
    tasks: list[dict] = []
    failed = 0

    skip_field = bool(os.environ.get("SKIP_FIELD_EXTRACTOR"))
    if not args.skip_inference:
        print(f"Running pipeline inference"
              f"{' (Textract skipped)' if skip_field else ''}...")

    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(range(len(candidates)), desc="Inference", unit="img")
    except ImportError:
        pbar = range(len(candidates))  # type: ignore

    for i in pbar:
        url = urls.get(i, "")
        if not url:
            failed += 1
            continue
        task: dict = {"data": {"image": url}}
        if not args.skip_inference:
            img_bytes = image_data.get(i)
            if img_bytes is not None:
                result = _run_pipeline(img_bytes)
                if result is not None:
                    task["predictions"] = _to_ls_predictions(result)
        tasks.append(task)

    # ── Step 5: write output ──────────────────────────────────────────────────
    out_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"\nBatch written: {out_path}  ({len(tasks)} tasks, {failed} failed, {size_kb:.0f} KB)")
    expiry_days = args.url_expiry // 86_400
    print(f"Presigned URLs valid for {expiry_days} day(s).")
    print(
        "\nNext steps:\n"
        f"  1. Label Studio → Import → select:  {out_path}\n"
        "  2. Skim predictions, correct wrong ones.\n"
        "  3. Export JSON → data/labels/label_studio_export.json\n"
        "  4. .\\scripts\\sync_to_cloud.ps1 -Message 'batch labels'\n"
        "  5. Colab: SYNC_IMAGES=True, REBUILD_DATASET=True → retrain."
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
