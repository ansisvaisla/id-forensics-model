"""Shared Colab setup helpers.

Architecture
────────────
Google Drive (persistent across sessions):
    My Drive/id-forensics/
        data/raw/id_doc_front_flat/   ← stage1/4 labeled images, synced from S3 once
        data/screen_images/           ← stage2 quality gate images, cached from S3 once
        outputs/                       ← trained weights
        eval/                            ← evaluation reports + viz images

GitHub (code + labels):
    scripts/, notebooks/, data/labels/label_studio_export.json, etc.

Colab VM (per-session, fast NVMe):
    /content/id-forensics-model/      ← git clone
    /content/id-forensics-model/data/raw  ← symlink → Drive raw images

Flow:
    1. cb.connect()              # Drive + GitHub + deps
    2. cb.sync_images_from_s3()  # optional — new images only
    3. cb.rebuild_splits()       # convert labels → YOLO train/val/test
    4. train / save_weights / run_eval

Colab Secrets required (🔑 left sidebar):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION
    AWS_SESSION_TOKEN          (only for temporary/SSO credentials — omit if using long-term keys)

    For batch labeling (section 7):
    ZENKA_KE_DATABASE_URL      (recommended — full postgresql:// URL)
    OR individually:
      ZENKA_KE_DB_HOST, ZENKA_KE_DB_USER, ZENKA_KE_DB_PASSWORD, ZENKA_KE_DB_NAME
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DRIVE_FOLDER = "id-forensics"
DRIVE_ROOT = Path(f"/content/drive/MyDrive/{DRIVE_FOLDER}")
DRIVE_RAW_DIR = DRIVE_ROOT / "data" / "raw" / "id_doc_front_flat"
DRIVE_SCREEN_DIR = DRIVE_ROOT / "data" / "screen_images"  # stage2 image cache
OUTPUTS_DIR = DRIVE_ROOT / "outputs"
EVAL_DIR = DRIVE_ROOT / "eval"
DRIVE_BATCHES_DIR = DRIVE_ROOT / "data" / "batches"
REPO_DIR = Path("/content/id-forensics-model")
GITHUB_USER = "ansisvaisla"
REPO_NAME = "id-forensics-model"

# S3 bucket and key prefix for labeled ID images
S3_BUCKET = "sf-zenka-ke-prod-media-svc"
S3_PREFIX = "id-doc-front/"

WEIGHT_TARGETS: dict[str, tuple[str, str]] = {
    "stage1": ("models/stage1_corners/weights/best.pt", "stage1_best.pt"),
    "stage2": ("models/stage2_screen/best.pt", "stage2_best.pt"),
    "stage4": ("models/stage4_id_type/best.pt", "stage4_best.pt"),
}


# ── GPU check ────────────────────────────────────────────────────────────────

def check_gpu() -> None:
    import torch
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


# ── Drive ────────────────────────────────────────────────────────────────────

def mount_drive() -> None:
    """Mount Google Drive at /content/drive and create folder structure."""
    from google.colab import drive  # type: ignore[import-untyped]
    drive.mount("/content/drive")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    DRIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Drive mounted: {DRIVE_ROOT}")


# ── GitHub ───────────────────────────────────────────────────────────────────

def _repo_url(github_token: str) -> str:
    if github_token:
        return f"https://{github_token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
    return f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"


def clone_or_pull(github_token: str = "") -> Path:
    """Clone repo if missing, otherwise git pull to get latest code + labels."""
    os.chdir("/content")
    bootstrap = REPO_DIR / "scripts" / "colab_bootstrap.py"

    if REPO_DIR.is_dir() and not bootstrap.is_file():
        print("Stale repo (missing colab_bootstrap.py) — removing and re-cloning...")
        subprocess.run(["rm", "-rf", str(REPO_DIR)], check=True)

    if not REPO_DIR.is_dir():
        url = _repo_url(github_token)
        safe = url.replace(github_token, "***") if github_token else url
        print(f"Cloning {safe} ...")
        result = subprocess.run(
            ["git", "clone", url, str(REPO_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed: {result.stderr.strip()}\n"
                "Private repo? Set GITHUB_TOKEN (github.com/settings/tokens, scope: repo)."
            )
        if not bootstrap.is_file():
            raise FileNotFoundError(
                "scripts/colab_bootstrap.py missing after clone.\n"
                "On work PC: git push origin main"
            )
    else:
        print("Pulling latest code and labels...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "origin", "main"], check=False)

    return REPO_DIR


# ── Dependencies ─────────────────────────────────────────────────────────────

def install_deps() -> None:
    """Install Python deps."""
    os.chdir(REPO_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "opencv-python-headless", "timm", "python-dotenv", "boto3"],
        check=True,
    )
    print("Dependencies installed.")


# ── Symlink ───────────────────────────────────────────────────────────────────

def _symlink_raw_to_drive() -> None:
    """Symlink REPO_DIR/data/raw/id_doc_front_flat -> Drive so scripts find images."""
    link = REPO_DIR / "data" / "raw" / "id_doc_front_flat"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == DRIVE_RAW_DIR.resolve():
            return  # already correct
        link.unlink()
    if link.is_dir():
        # Was a real directory (e.g. from a previous session with local images)
        # Move any files that aren't on Drive yet, then replace with symlink
        for f in link.iterdir():
            dest = DRIVE_RAW_DIR / f.name
            if not dest.exists():
                f.rename(dest)
        import shutil
        shutil.rmtree(str(link))
    link.symlink_to(DRIVE_RAW_DIR)
    print(f"Symlink: {link} -> {DRIVE_RAW_DIR}")


# ── S3 sync ───────────────────────────────────────────────────────────────────

def _s3_secret(name: str, required: bool = True) -> str:
    """Read from Colab Secrets. Returns empty string if not required and missing."""
    try:
        from google.colab import userdata  # type: ignore[import-untyped]
        val = userdata.get(name)
        return val or ""
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"Colab Secret '{name}' not found.\n"
                "Add it: left sidebar → 🔑 Secrets → New secret.\n"
                f"  ({exc})"
            ) from exc
        return ""


def _labeled_s3_keys() -> list[str]:
    """Return S3 keys for all labeled images in the Label Studio export."""
    import json

    export = REPO_DIR / "data" / "labels" / "label_studio_export.json"
    tasks = json.loads(export.read_text(encoding="utf-8"))

    keys: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        anns = task.get("annotations") or []
        if not any(not a.get("was_cancelled", False) for a in anns):
            continue

        # Path 1: task imported via s3:// URI (batch_label.py --s3-uris)
        image_uri: str = task.get("data", {}).get("image", "")
        if image_uri.startswith("s3://"):
            without_prefix = image_uri.removeprefix("s3://")
            _, _, s3_key = without_prefix.partition("/")
            if s3_key and s3_key not in seen:
                seen.add(s3_key)
                keys.append(s3_key)
            continue

        # Path 2: task uploaded directly to Label Studio (file_upload field)
        fu = task.get("file_upload", "")
        flat = fu.split("-", 1)[1] if "-" in fu else fu
        if not flat or flat in seen:
            continue
        seen.add(flat)
        stem, ext = flat.rsplit(".", 1) if "." in flat else (flat, "jpg")
        parts = stem.split("_")
        if len(parts) >= 4 and len(parts[0]) == 4 and parts[0].isdigit():
            year, month, day = parts[0], parts[1], parts[2]
            fname = "_".join(parts[3:])
            keys.append(f"{S3_PREFIX}{year}/{month}/{day}/{fname}.{ext}")
        else:
            keys.append(f"{S3_PREFIX}{flat}")
    return keys


def sync_images_from_s3(workers: int = 16) -> int:
    """Download labeled images from S3 → Drive (skips files already on Drive).

    Reads AWS credentials from Colab Secrets.
    AWS_SESSION_TOKEN is optional — only needed for temporary/SSO credentials.

    Returns number of newly downloaded files.
    """
    import boto3  # type: ignore[import-untyped]
    from concurrent.futures import ThreadPoolExecutor, as_completed

    key_id = _s3_secret("AWS_ACCESS_KEY_ID")
    secret = _s3_secret("AWS_SECRET_ACCESS_KEY")
    region = _s3_secret("AWS_DEFAULT_REGION")
    token = _s3_secret("AWS_SESSION_TOKEN", required=False)

    session = boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        aws_session_token=token or None,
        region_name=region,
    )
    from botocore.config import Config  # type: ignore[import-untyped]
    s3 = session.client(
        "s3",
        config=Config(max_pool_connections=max(workers, 10)),
    )

    keys = _labeled_s3_keys()
    print(f"Labeled images in export: {len(keys)}")

    DRIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)

    def _download_one(key: str) -> str:
        flat = key.replace(S3_PREFIX, "").replace("/", "_")
        dest = DRIVE_RAW_DIR / flat
        if dest.is_file():
            return "skipped"
        s3.download_file(S3_BUCKET, key, str(dest))
        return "downloaded"

    downloaded = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, k): k for k in keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                status = future.result()
                if status == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"  FAIL {key}: {exc}")

    total = len(list(DRIVE_RAW_DIR.glob("*.jpg")))
    print(f"S3 sync: {downloaded} new, {skipped} cached, {failed} failed")
    print(f"Drive images: {total}  ({DRIVE_RAW_DIR})")
    if failed:
        print(f"WARNING: {failed} images failed to download.")
    return downloaded


# ── Convert + split ───────────────────────────────────────────────────────────

def rebuild_splits(skip_screen: bool = False) -> None:
    """Run convert_labels_to_yolo + split_yolo_dataset on the Colab VM.

    Images are read via the symlink REPO_DIR/data/raw/ → Drive.
    YOLO splits are written to REPO_DIR/data/yolo/ (VM fast disk — not Drive).

    Args:
        skip_screen: pass True when the quality gate dataset has already been
                     built by convert_labels_ls_to_quality_gate.py so that
                     convert_labels_to_yolo does not overwrite it with the
                     old 3-class screen labels.
    """
    os.chdir(REPO_DIR)
    print("Converting labels to YOLO format...")
    cmd = [sys.executable, "scripts/convert_labels_to_yolo.py"]
    if skip_screen:
        cmd.append("--no-screen")
    # Pass the Drive screen_images cache as an extra image root so that
    # good_front/partial images downloaded by convert_labels_ls_to_quality_gate.py
    # are found by the corners converter (they live in DRIVE_SCREEN_DIR, not data/raw/).
    if DRIVE_SCREEN_DIR.is_dir():
        cmd += ["--extra-image-roots", str(DRIVE_SCREEN_DIR)]
    subprocess.run(cmd, check=True, cwd=REPO_DIR)
    print("Building train/val/test splits...")
    dataset = "corners" if skip_screen else "both"
    subprocess.run(
        [sys.executable, "scripts/split_yolo_dataset.py", "--dataset", dataset],
        check=True, cwd=REPO_DIR,
    )


# ── YOLO cache ────────────────────────────────────────────────────────────────

def clear_corner_caches() -> None:
    """Delete YOLO label .cache files (safe after label format changes)."""
    os.chdir(REPO_DIR)
    for cache in (REPO_DIR / "data" / "yolo" / "corners" / "labels").rglob("*.cache"):
        cache.unlink()
        print("deleted", cache)


# ── Connect (infra only — no data processing) ────────────────────────────────

def connect(github_token: str = "") -> Path:
    """Mount Drive, clone/pull GitHub, install deps, symlink images.

    Does NOT download from S3 or convert labels — run those separately.
    """
    mount_drive()
    clone_or_pull(github_token=github_token)
    install_deps()
    _symlink_raw_to_drive()
    return REPO_DIR


# ── Legacy convenience wrapper ─────────────────────────────────────────────────

def setup(
    github_token: str = "",
    sync_images: bool = True,
    workers: int = 16,
) -> Path:
    """Full pipeline: connect → optional S3 sync → rebuild splits."""
    connect(github_token=github_token)
    if sync_images:
        sync_images_from_s3(workers=workers)
    else:
        n = len(list(DRIVE_RAW_DIR.glob("*.jpg")))
        print(f"Skipping S3 sync — {n} images already on Drive.")
    rebuild_splits()
    return REPO_DIR


# ── Weight management ─────────────────────────────────────────────────────────

def restore_weights(stage: str) -> bool:
    """Copy weights from Drive back into the local repo.

    Returns True if weights are now present, False if not found on Drive.
    Safe to call even if weights are already present (no-op in that case).
    """
    if stage not in WEIGHT_TARGETS:
        raise ValueError(f"Unknown stage {stage!r}. Choose from: {list(WEIGHT_TARGETS)}")

    local_rel, drive_name = WEIGHT_TARGETS[stage]
    dst = REPO_DIR / local_rel
    if dst.is_file():
        print(f"{stage}: weights already present at {dst}")
        return True

    src = OUTPUTS_DIR / drive_name
    if not src.is_file():
        print(f"WARNING: {stage} weights not found on Drive ({src}). Train this stage first.")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src, dst)
    print(f"{stage}: restored {drive_name} from Drive ({dst.stat().st_size / 1e6:.1f} MB)")
    return True


def restore_all_weights() -> None:
    """Restore all available stage weights from Drive. Prints a summary."""
    for stage in WEIGHT_TARGETS:
        restore_weights(stage)


def save_weights(stage: str) -> Path:
    """Copy trained weights to Drive outputs/. Returns Drive destination path."""
    if stage not in WEIGHT_TARGETS:
        raise ValueError(f"Unknown stage {stage!r}. Choose from: {list(WEIGHT_TARGETS)}")

    local_rel, drive_name = WEIGHT_TARGETS[stage]
    src = REPO_DIR / local_rel
    if not src.is_file():
        raise FileNotFoundError(f"Weights not found: {src}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUTPUTS_DIR / drive_name
    import shutil
    shutil.copy2(src, dst)
    print(f"Saved {drive_name} to Drive ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def run_eval(stage: str, split: str = "val") -> Path:
    """Run evaluate_models.py and save all outputs to Drive (persistent).

    Returns the eval run directory on Drive.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_DIR)
    cmd = [
        sys.executable,
        "scripts/evaluate_models.py",
        "--stage", stage,
        "--split", split,
        "--output-dir", str(EVAL_DIR),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_DIR)

    # Find the latest run dir for this stage
    stage_dirs = sorted((EVAL_DIR / stage).glob("*"), reverse=True)
    if not stage_dirs:
        raise FileNotFoundError(f"No eval output found under {EVAL_DIR / stage}")
    latest = stage_dirs[0]
    print(f"Eval saved to Drive: {latest}")
    print(f"  wrong images: {latest / 'viz' / 'wrong'}")
    print(f"  report:       {latest / 'report.txt'}")
    return latest


# ── Batch labeling ─────────────────────────────────────────────────────────────

def run_batch_label(
    limit: int = 1000,
    hours: int = 720,
    skip_inference: bool = False,
    url_expiry: int = 604_800,
) -> Path:
    """Generate a Label Studio import JSON pre-annotated by the pipeline.

    DB credentials are read automatically from Colab Secrets:
      • ZENKA_KE_DATABASE_URL  (recommended — full postgresql:// URL)
      • OR individual: ZENKA_KE_DB_HOST, ZENKA_KE_DB_USER, ZENKA_KE_DB_PASSWORD,
                       ZENKA_KE_DB_NAME, ZENKA_KE_DB_PORT

    AWS credentials (for presigned URLs + image download) come from the same
    Colab Secrets used for S3 sync: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION, AWS_SESSION_TOKEN (optional).

    Output is saved to Drive: My Drive/id-forensics/data/batches/<timestamp>_batch.json
    Returns the Path to the written file.
    """
    DRIVE_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_DIR)

    # Ensure DB secrets from Colab Secrets are injected into the environment
    # so the subprocess can read them (google.colab.userdata is only available
    # in the notebook process).
    try:
        from google.colab import userdata  # type: ignore[import-untyped]

        # Build a clean DSN from individual parts (overwrites any stale DSN).
        host = (userdata.get("ZENKA_KE_DB_HOST") or "").strip()
        user = (userdata.get("ZENKA_KE_DB_USER") or "").strip()
        pwd = (userdata.get("ZENKA_KE_DB_PASSWORD") or "").strip()
        name = (userdata.get("ZENKA_KE_DB_NAME") or "").strip()
        port = (userdata.get("ZENKA_KE_DB_PORT") or "").strip()

        if host and user and pwd and name:
            from urllib.parse import quote
            user_e = quote(user, safe="")
            pwd_e = quote(pwd, safe="")
            host_part = f"{host}:{port}" if port else host
            dsn = f"postgresql://{user_e}:{pwd_e}@{host_part}/{name}"
            os.environ["ZENKA_KE_DATABASE_URL"] = dsn
            print("Injected Colab secrets: built ZENKA_KE_DATABASE_URL from individual parts")
        else:
            # Try individual DB parts
            host = userdata.get("ZENKA_KE_DB_HOST") or ""
            user = userdata.get("ZENKA_KE_DB_USER") or ""
            pwd = userdata.get("ZENKA_KE_DB_PASSWORD") or ""
            name = userdata.get("ZENKA_KE_DB_NAME") or ""
            port = userdata.get("ZENKA_KE_DB_PORT") or ""
            if host and user and pwd and name:
                os.environ.setdefault("ZENKA_KE_DB_HOST", host)
                os.environ.setdefault("ZENKA_KE_DB_USER", user)
                os.environ.setdefault("ZENKA_KE_DB_PASSWORD", pwd)
                os.environ.setdefault("ZENKA_KE_DB_NAME", name)
                if port:
                    os.environ.setdefault("ZENKA_KE_DB_PORT", port)
                print("Injected Colab secrets: ZENKA_KE_DB_*")
    except Exception:
        # Not running in Colab or userdata unavailable — fall back to .env
        pass

    cmd = [
        sys.executable,
        "scripts/batch_label.py",
        "--limit",
        str(limit),
        "--hours",
        str(hours),
        "--url-expiry",
        str(url_expiry),
    ]
    if skip_inference:
        cmd.append("--skip-inference")

    try:
        # Capture output so we can show a helpful error in the notebook if it fails
        proc = subprocess.run(cmd, check=True, cwd=REPO_DIR, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout)
    except subprocess.CalledProcessError as exc:
        # Show stderr to help debugging inside the Colab cell
        err = exc.stderr or str(exc)
        print("Batch label script failed with return code", exc.returncode)
        print("=== STDERR ===")
        print(err)
        print("=== STDOUT ===")
        print(exc.stdout or "")
        raise RuntimeError(f"batch_label.py failed (rc={exc.returncode})\n{err}") from exc

    # Find the most-recent batch JSON written to Drive
    batch_files = sorted(DRIVE_BATCHES_DIR.glob("*_batch.json"), reverse=True)
    if not batch_files:
        raise FileNotFoundError(f"No batch JSON found under {DRIVE_BATCHES_DIR}")
    latest = batch_files[0]
    print(f"\nBatch saved to Drive: {latest}")
    print("Import into Label Studio: open project → Import → select that file.")
    return latest
