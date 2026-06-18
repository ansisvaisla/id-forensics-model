"""Shared Colab setup helpers — import after cloning the repo on a Colab runtime.

Drive layout (create once in Google Drive):
    My Drive/id-forensics/
        id_forensics_home_data.zip
        outputs/
            stage1_best.pt
            stage2_best.pt
            stage4_best.pt

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


def check_gpu() -> None:
    """Print CUDA availability and GPU name."""
    import torch

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def mount_drive() -> None:
    """Mount Google Drive at /content/drive."""
    from google.colab import drive

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


def install_deps() -> None:
    """Install Python deps and run verify_home_setup."""
    os.chdir(REPO_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "opencv-python-headless", "timm", "python-dotenv"],
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
