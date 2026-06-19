"""Shared Colab setup helpers — import after cloning the repo on a Colab runtime.

Drive layout (create once in Google Drive):
    My Drive/id-forensics/
        id_forensics_home_data.zip   ← fallback: pack_for_home.py output
        outputs/
            stage1_best.pt
            stage2_best.pt
            stage4_best.pt

AWS S3 setup (preferred for images):
    Store four Colab Secrets (🔑 in left sidebar):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
        AWS_DEFAULT_REGION, S3_BUCKET_NAME
    Then call cb.download_from_s3() instead of cb.extract_data().

Usage in a notebook (after clone):
    import sys
    sys.path.insert(0, "/content/id-forensics-model/scripts")
    import colab_bootstrap as cb
    cb.setup_workspace(github_token=GITHUB_TOKEN)
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ── Paths (keep in sync with scripts/sync_*.ps1) ─────────────────────────────
DRIVE_FOLDER = "id-forensics"
DRIVE_ROOT = Path(f"/content/drive/MyDrive/{DRIVE_FOLDER}")
ZIP_PATH = DRIVE_ROOT / "id_forensics_home_data.zip"
OUTPUTS_DIR = DRIVE_ROOT / "outputs"
REPO_DIR = Path("/content/id-forensics-model")
GITHUB_USER = "ansisvaisla"
REPO_NAME = "id-forensics-model"

WEIGHT_TARGETS: dict[str, tuple[str, str]] = {
    "stage1": ("models/stage1_corners/weights/best.pt", "stage1_best.pt"),
    "stage2": ("models/stage2_screen/best.pt", "stage2_best.pt"),
    "stage4": ("models/stage4_id_type/best.pt", "stage4_best.pt"),
}

# S3 prefixes to sync down (relative to bucket root → local data/)
_S3_SYNC_TARGETS: list[tuple[str, str]] = [
    ("data/raw/", "data/raw/"),
    ("data/labels/", "data/labels/"),
]


def check_gpu() -> None:
    """Print CUDA availability and GPU name."""
    import torch

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def mount_drive() -> None:
    """Mount Google Drive at /content/drive."""
    from google.colab import drive  # type: ignore[import-untyped]

    drive.mount("/content/drive")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Drive root: {DRIVE_ROOT}")


def _repo_url(github_token: str) -> str:
    if github_token:
        return f"https://{github_token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
    return f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"


def _bootstrap_path() -> Path:
    return REPO_DIR / "scripts" / "colab_bootstrap.py"


def clone_repo(github_token: str = "", force: bool = False) -> Path:
    """Clone or refresh GitHub repo at REPO_DIR. Returns repo path."""
    os.chdir("/content")
    bootstrap = _bootstrap_path()

    if REPO_DIR.is_dir() and not bootstrap.is_file():
        print("Stale repo (missing colab_bootstrap.py) — removing and re-cloning...")
        force = True

    if REPO_DIR.is_dir():
        if force:
            subprocess.run(["rm", "-rf", str(REPO_DIR)], check=True)
        else:
            print(f"Repo exists: {REPO_DIR}")
            return REPO_DIR

    url = _repo_url(github_token)
    safe = url.replace(github_token, "***") if github_token else url
    print(f"Cloning {safe} ...")
    result = subprocess.run(
        ["git", "clone", url, str(REPO_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "git clone failed.\n"
            f"stderr: {result.stderr.strip()}\n\n"
            "Private repo: set GITHUB_TOKEN (https://github.com/settings/tokens, scope: repo)."
        )
    if not bootstrap.is_file():
        raise FileNotFoundError(
            "scripts/colab_bootstrap.py missing after clone.\n"
            "On work PC: git push origin main (bootstrap must be on GitHub)."
        )
    return REPO_DIR


def resolve_zip_path() -> Path:
    """Find training zip on Drive (new folder layout or legacy root)."""
    legacy = Path("/content/drive/MyDrive/id_forensics_home_data.zip")
    for candidate in (ZIP_PATH, legacy):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Zip not found. Tried:\n  {ZIP_PATH}\n  {legacy}\n"
        f"Upload to: My Drive/{DRIVE_FOLDER}/id_forensics_home_data.zip"
    )


def extract_data() -> None:
    """Unzip training data from Drive into the repo root."""
    zip_path = resolve_zip_path()
    print(f"Using zip: {zip_path}")
    os.chdir(REPO_DIR)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")
    n_corners = len(list(Path("data/yolo/corners/images/train").glob("*.jpg")))
    n_screen = len(list(Path("data/yolo/screen/images/train").glob("*.jpg")))
    n_id_type = len(list(Path("data/id_type/train").rglob("*.jpg")))
    print(f"Repo ready: {os.getcwd()}")
    print(f"  corners train images: {n_corners}")
    print(f"  screen train images:  {n_screen}")
    print(f"  id_type train images: {n_id_type}")


def _get_s3_secret(name: str) -> str:
    """Read a secret from Colab Secrets. Raises clear error if missing."""
    try:
        from google.colab import userdata  # type: ignore[import-untyped]
        return userdata.get(name)
    except Exception as exc:
        raise RuntimeError(
            f"Colab Secret '{name}' not found.\n"
            "Add it in Colab: left sidebar → 🔑 Secrets → New secret.\n"
            f"  ({exc})"
        ) from exc


def download_from_s3(
    rebuild_splits: bool = True,
    run_convert: bool = True,
) -> None:
    """Pull raw images + label export directly from S3 using Colab Secrets.

    Reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION,
    S3_BUCKET_NAME from Colab Secrets — credentials never appear in the notebook.

    Args:
        rebuild_splits: Re-run convert_labels + split scripts after download.
        run_convert:    Run convert_labels_to_yolo.py (set False to skip if labels
                        are already up-to-date).
    """
    import boto3  # type: ignore[import-untyped]

    key_id = _get_s3_secret("AWS_ACCESS_KEY_ID")
    secret = _get_s3_secret("AWS_SECRET_ACCESS_KEY")
    region = _get_s3_secret("AWS_DEFAULT_REGION")
    bucket = _get_s3_secret("S3_BUCKET_NAME")

    session = boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=region,
    )
    s3 = session.client("s3")

    os.chdir(REPO_DIR)
    total = 0
    for s3_prefix, local_prefix in _S3_SYNC_TARGETS:
        local_dir = REPO_DIR / local_prefix
        local_dir.mkdir(parents=True, exist_ok=True)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(s3_prefix):]
                if not rel:
                    continue
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.stat().st_size != obj["Size"]:
                    s3.download_file(bucket, key, str(dest))
                    total += 1
        print(f"  s3://{bucket}/{s3_prefix}  ->  {local_prefix}  ({total} new files)")

    print(f"S3 sync done. {total} files downloaded.")

    if run_convert:
        print("Running convert_labels_to_yolo.py ...")
        subprocess.run(
            [sys.executable, "scripts/convert_labels_to_yolo.py"],
            check=True,
            cwd=REPO_DIR,
        )

    if rebuild_splits:
        print("Rebuilding train/val/test splits ...")
        subprocess.run(
            [sys.executable, "scripts/split_yolo_dataset.py", "--dataset", "both"],
            check=True,
            cwd=REPO_DIR,
        )
        subprocess.run(
            [sys.executable, "scripts/split_id_type_dataset.py", "--source", "all_deskewed"],
            check=True,
            cwd=REPO_DIR,
        )


def install_deps() -> None:
    """Install Python deps and run verify_home_setup."""
    os.chdir(REPO_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "opencv-python-headless", "timm", "python-dotenv", "boto3"],
        check=True,
    )
    subprocess.run([sys.executable, "scripts/verify_home_setup.py"], check=True)


def clear_corner_caches() -> None:
    """Delete YOLO label .cache files (safe after label format changes)."""
    os.chdir(REPO_DIR)
    labels_root = Path("data/yolo/corners/labels")
    if not labels_root.is_dir():
        return
    for cache in labels_root.rglob("*.cache"):
        cache.unlink()
        print("deleted", cache)


def download_from_colab_manifest(
    manifest_path: Path | str,
    rebuild_splits: bool = True,
) -> None:
    """Download training images via pre-signed HTTPS URLs — no AWS credentials in Colab.

    Work PC generates the manifest with:
        python scripts/prepare_training_data.py --write-colab-manifest

    Upload the small JSON file to Colab (Files panel) or Drive, then call this.
    """
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}\n"
            "On work PC run:\n"
            "  python scripts/prepare_training_data.py --write-colab-manifest\n"
            "Then upload data/manifests/colab_presigned.json to Colab."
        )
    os.chdir(REPO_DIR)
    cmd = [
        sys.executable,
        "scripts/prepare_training_data.py",
        "--from-colab-manifest",
        str(manifest),
    ]
    if not rebuild_splits:
        cmd.append("--skip-pipeline")
    subprocess.run(cmd, check=True, cwd=REPO_DIR)


def setup_workspace(github_token: str = "", remount_drive: bool = True) -> Path:
    """Full Colab setup: mount Drive, clone, extract, install. Idempotent."""
    if remount_drive:
        mount_drive()
    clone_repo(github_token=github_token)
    extract_data()
    install_deps()
    return REPO_DIR


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
    print(f"Saved to Drive: {dst}")
    print(f"Size: {dst.stat().st_size / 1e6:.1f} MB")
    return dst
