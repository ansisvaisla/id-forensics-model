"""Batch labeling loop — generate a Label Studio import JSON with pipeline pre-annotations.

Workflow
────────
1. Query DB for recent unlabeled ID-front images (postgres2).
2. Filter out images already in label_studio_export.json.
3. For each candidate:
   a. Generate a presigned S3 URL (LS fetches the image — nothing downloaded permanently).
   b. Run the pipeline (download temporarily, run stages 1/2/4, discard bytes).
   c. Map pipeline output → Label Studio predictions (pre-filled choices).
4. Write  data/batches/<YYYYMMDD_HHMMSS>_batch.json
5. In Label Studio: open project → Import → select the JSON file.
6. Skim the pre-annotated tasks, fix wrong predictions.
7. Export JSON → overwrite  data/labels/label_studio_export.json
8. Run  .\\scripts\\sync_to_cloud.ps1  → retrain on Colab.

Usage
─────
    python scripts/batch_label.py
    python scripts/batch_label.py --limit 1000 --hours 720
    python scripts/batch_label.py --limit 200  --skip-inference   # no pipeline, faster
    python scripts/batch_label.py --out data/batches/my_batch.json

Notes
─────
- Presigned URLs expire after --url-expiry seconds (default 604800 = 7 days).
  Generate a fresh batch if you haven't imported within that window.
- Pipeline models must be present under models/ on this machine.
  If a model is missing the task is still written; only predictions are omitted.
- DB credentials come from .env (ZENKA_KE_DB_*).
- AWS credentials come from .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_FILE = PROJECT_ROOT / "data" / "labels" / "label_studio_export.json"
BATCHES_DIR = PROJECT_ROOT / "data" / "batches"

# Google Drive path (used automatically when running in Colab)
_DRIVE_BATCHES_DIR = Path("/content/drive/MyDrive/id-forensics/data/batches")

# Pipeline version tag written into predictions.model_version
_MODEL_VERSION = "id-forensics-pipeline"

def _inject_colab_db_secrets() -> bool:
    """Read DB credentials from Colab Secrets and inject into os.environ.

    Supports two modes:
      • Single secret:    ZENKA_KE_DATABASE_URL  (full postgresql:// URL)
      • Individual parts: ZENKA_KE_DB_HOST, ZENKA_KE_DB_USER,
                          ZENKA_KE_DB_PASSWORD, ZENKA_KE_DB_NAME, ZENKA_KE_DB_PORT

    Returns True if running in Colab (even if no secrets found).
    """
    try:
        from google.colab import userdata  # type: ignore[import-untyped]
    except ImportError:
        return False  # not in Colab

    def _get(name: str) -> str:
        try:
            return userdata.get(name) or ""
        except Exception:
            return ""

    url = _get("ZENKA_KE_DATABASE_URL")
    if url:
        os.environ.setdefault("ZENKA_KE_DATABASE_URL", url)
        print("DB credentials loaded from Colab Secret: ZENKA_KE_DATABASE_URL")
        return True

    host = _get("ZENKA_KE_DB_HOST")
    user = _get("ZENKA_KE_DB_USER")
    password = _get("ZENKA_KE_DB_PASSWORD")
    name = _get("ZENKA_KE_DB_NAME")
    port = _get("ZENKA_KE_DB_PORT") or "5432"

    if host and user and password and name:
        os.environ.setdefault("ZENKA_KE_DB_HOST", host)
        os.environ.setdefault("ZENKA_KE_DB_USER", user)
        os.environ.setdefault("ZENKA_KE_DB_PASSWORD", password)
        os.environ.setdefault("ZENKA_KE_DB_NAME", name)
        os.environ.setdefault("ZENKA_KE_DB_PORT", port)
        print("DB credentials loaded from Colab Secrets (individual vars)")
        return True

    print(
        "WARNING: No DB secrets found in Colab.\n"
        "  Add one of these in the 🔑 Secrets sidebar:\n"
        "    • ZENKA_KE_DATABASE_URL  (recommended — full postgresql:// URL)\n"
        "    • OR: ZENKA_KE_DB_HOST, ZENKA_KE_DB_USER, ZENKA_KE_DB_PASSWORD, ZENKA_KE_DB_NAME"
    )
    return True


def _default_batches_dir() -> Path:
    """Return Drive batches dir in Colab, local data/batches otherwise."""
    if _DRIVE_BATCHES_DIR.parent.parent.is_dir():  # Drive is mounted
        return _DRIVE_BATCHES_DIR
    return BATCHES_DIR




def _flat_name(file_upload: str) -> str:
    """Strip Label Studio UUID prefix: 'abc123-2023_05_18_xyz.jpg' → '2023_05_18_xyz.jpg'."""
    parts = file_upload.split("-", 1)
    return parts[1] if len(parts) == 2 else file_upload


def _already_labeled(export_path: Path) -> set[str]:
    """Return set of flat filenames (stems) already present in the label export."""
    if not export_path.is_file():
        return set()
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    stems: set[str] = set()
    for task in tasks:
        fu = task.get("file_upload", "")
        if fu:
            stems.add(Path(_flat_name(fu)).stem)
    return stems


# ── S3 key ↔ flat filename ────────────────────────────────────────────────────

def _flat_from_s3_key(s3_key: str) -> str:
    """id-doc-front/2023/11/22/abc123.jpg → 2023_11_22_abc123.jpg"""
    prefix = "id-doc-front/"
    rel = s3_key.removeprefix(prefix).lstrip("/")
    return rel.replace("/", "_")


def _stem_from_s3_key(s3_key: str) -> str:
    return Path(_flat_from_s3_key(s3_key)).stem


# ── Candidate fetch ───────────────────────────────────────────────────────────

def _fetch_candidates(
    hours: int,
    limit: int,
    labeled_stems: set[str],
) -> list:
    """Fetch recent ID_DOC_FRONT attachments not yet in the label export."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from db_zenka_ke import fetch_attachments_with_s3_keys, SUB_TYPE_ID_FRONT

    # Fetch extra to compensate for overlap with already-labeled
    rows = fetch_attachments_with_s3_keys(
        SUB_TYPE_ID_FRONT,
        limit=limit * 3,
        hours=hours,
    )
    unseen = [r for r in rows if _stem_from_s3_key(r.s3_key) not in labeled_stems]
    return unseen[:limit]


# ── Presigned URL ─────────────────────────────────────────────────────────────

def _presigned_url(s3_key: str, bucket: str, expiry: int) -> str:
    """Generate a temporary HTTPS URL for Label Studio image display."""
    import boto3
    from botocore.config import Config

    client = boto3.client("s3", config=Config(signature_version="s3v4"))
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expiry,
    )


# ── Pipeline inference ────────────────────────────────────────────────────────

def _download_bytes(s3_key: str, bucket: str) -> Optional[bytes]:
    """Download image bytes from S3 without writing to disk."""
    try:
        import boto3

        client = boto3.client("s3")
        buf = io.BytesIO()
        client.download_fileobj(bucket, s3_key, buf)
        return buf.getvalue()
    except Exception as exc:
        print(f"  [download] FAIL {s3_key}: {exc}", file=sys.stderr)
        return None


def _run_pipeline(image_bytes: bytes) -> Optional[object]:
    """Run orchestration.run() on raw image bytes. Returns PipelineResult or None."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import orchestration

        return orchestration.run(image_bytes)
    except Exception as exc:
        print(f"  [pipeline] FAIL: {exc}", file=sys.stderr)
        return None


# ── LS prediction builder ─────────────────────────────────────────────────────

def _quality_from_result(result) -> str:
    """Map PipelineResult to a quality label string."""
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
    """Convert PipelineResult to Label Studio predictions array."""
    quality = _quality_from_result(result)
    ls_results = [
        {
            "from_name": "quality",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [quality]},
        },
    ]

    id_type_label: str = "unknown_id"
    if result.id_type is not None:
        raw = result.id_type.id_type
        id_type_label = raw if raw != "unknown" else "unknown_id"

    ls_results.append(
        {
            "from_name": "id_type",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [id_type_label]},
        }
    )

    # Confidence: use presentation attack score when it's an attack, else id_type confidence
    if result.is_screen_replay and result.presentation_attack:
        confidence = result.presentation_attack.screen_score
    elif result.is_printout and result.presentation_attack:
        confidence = result.presentation_attack.printout_score
    elif result.id_type is not None:
        confidence = result.id_type.confidence
    else:
        confidence = 0.5

    return [
        {
            "model_version": _MODEL_VERSION,
            "score": round(float(confidence), 4),
            "result": ls_results,
        }
    ]


# ── LS task builder ───────────────────────────────────────────────────────────

def _build_task(
    s3_key: str,
    bucket: str,
    url_expiry: int,
    run_inference: bool,
) -> dict:
    """Build one Label Studio task dict for a single image."""
    url = _presigned_url(s3_key, bucket, url_expiry)
    task: dict = {"data": {"image": url}}

    if run_inference:
        image_bytes = _download_bytes(s3_key, bucket)
        if image_bytes is not None:
            result = _run_pipeline(image_bytes)
            if result is not None:
                task["predictions"] = _to_ls_predictions(result)

    return task


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # Inject DB creds from Colab Secrets first, then fall back to .env
    _inject_colab_db_secrets()
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Generate Label Studio import JSON with pipeline pre-annotations"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max new images per batch (default: 1000)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=720,
        help="Look back this many hours in DB (default: 720 = 30 days)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. "
            "Default: Drive data/batches/ in Colab, or local data/batches/ otherwise."
        ),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip pipeline — generate tasks without predictions (much faster)",
    )
    parser.add_argument(
        "--url-expiry",
        type=int,
        default=604_800,
        help="Presigned URL lifetime in seconds (default: 604800 = 7 days)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=EXPORT_FILE,
        help="Path to existing label export (for deduplication)",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (_default_batches_dir() / f"{ts}_batch.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load already-labeled stems ───────────────────────────────────
    labeled_stems = _already_labeled(args.export)
    print(f"Already labeled: {len(labeled_stems)} images")

    # ── Step 2: fetch candidates ─────────────────────────────────────────────
    print(f"Querying DB for recent ID_DOC_FRONT images (last {args.hours}h)...")
    try:
        candidates = _fetch_candidates(args.hours, args.limit, labeled_stems)
    except SystemExit as exc:
        print(f"DB connection failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"DB query failed: {exc}", file=sys.stderr)
        return 1

    if not candidates:
        print("No new unlabeled candidates found. Nothing to do.")
        return 0

    print(f"Candidates to label: {len(candidates)}")
    if not args.skip_inference:
        print("Running pipeline inference on each image (use --skip-inference to skip)...")
    else:
        print("Skipping inference (--skip-inference). Tasks will have no predictions.")

    # ── Step 3: build tasks ───────────────────────────────────────────────────
    tasks: list[dict] = []
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(candidates, desc="Building tasks", unit="img")
    except ImportError:
        iterator = candidates  # type: ignore
        print("(install tqdm for a progress bar: pip install tqdm)")

    failed = 0
    for row in iterator:
        try:
            task = _build_task(
                s3_key=row.s3_key,
                bucket=row.bucket,
                url_expiry=args.url_expiry,
                run_inference=not args.skip_inference,
            )
            tasks.append(task)
        except Exception as exc:
            failed += 1
            print(f"  FAIL {row.s3_key}: {exc}", file=sys.stderr)

    # ── Step 4: write output ──────────────────────────────────────────────────
    out_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"\nBatch written: {out_path}  ({len(tasks)} tasks, {failed} failed, {size_kb:.0f} KB)")
    if args.url_expiry < 86_400:
        print(f"WARNING: URLs expire in {args.url_expiry // 3600}h — import soon.")
    else:
        expiry_days = args.url_expiry // 86_400
        print(f"Presigned URLs valid for {expiry_days} day(s).")

    print(
        "\nNext steps:\n"
        f"  1. Label Studio → open project → Import → select:\n"
        f"       {out_path}\n"
        "  2. Skim the pre-annotated tasks, correct wrong predictions.\n"
        "  3. Export JSON → save as  data/labels/label_studio_export.json\n"
        "  4. Run:  .\\scripts\\sync_to_cloud.ps1 -Message 'batch labels'\n"
        "  5. Colab workbench: SYNC_IMAGES=True, REBUILD_DATASET=True → retrain."
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
