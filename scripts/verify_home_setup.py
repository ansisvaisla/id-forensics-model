"""Verify the environment is ready for training on a home GPU machine.

Usage:
    python scripts/verify_home_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> int:
    print("=== Home training setup check ===\n")
    all_ok = True

    # Python modules
    try:
        import torch  # noqa: F401
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU only"
        all_ok &= _check("torch", True, f"CUDA={cuda} ({device})")
    except ImportError:
        all_ok &= _check("torch", False, "pip install -r requirements.txt")

    for mod in ("torchvision", "ultralytics", "cv2", "PIL"):
        try:
            __import__(mod)
            _check(mod, True)
        except ImportError:
            all_ok &= _check(mod, False, "missing — run pip install -r requirements.txt")

    # Data layout
    yolo_screen_train = PROJECT_ROOT / "data" / "yolo" / "screen" / "images" / "train"
    n_train = len(list(yolo_screen_train.glob("*.jpg"))) if yolo_screen_train.is_dir() else 0
    all_ok &= _check(
        "data/yolo/screen train images",
        n_train > 0,
        f"{n_train} images" if n_train else "empty — extract id_forensics_home_data.zip",
    )

    yolo_corners_train = PROJECT_ROOT / "data" / "yolo" / "corners" / "images" / "train"
    n_corners = len(list(yolo_corners_train.glob("*.jpg"))) if yolo_corners_train.is_dir() else 0
    _check(
        "data/yolo/corners train images",
        n_corners > 0,
        f"{n_corners} images" if n_corners else "optional for screen-only training",
    )

    labels = PROJECT_ROOT / "data" / "labels" / "label_studio_export.json"
    all_ok &= _check("label export JSON", labels.is_file(), str(labels.name))

    # Models optional
    s2 = PROJECT_ROOT / "models" / "stage2_screen" / "best.pt"
    _check("stage2_screen weights", s2.is_file(), "optional — will train from scratch if missing")

    print()
    if all_ok:
        print("Ready to train. Run:")
        print("  python scripts/training/train_stage2_screen.py")
        return 0
    print("Fix FAIL items above before training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
