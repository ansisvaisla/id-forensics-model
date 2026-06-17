"""Stage 3 — Injection & Digital Tampering Detection.

Entry point: run(image_bytes) -> TamperingResult

No ML model required for v1 — fully algorithmic:
  1. ELA (Error Level Analysis): re-save at known JPEG quality, diff against original.
  2. EXIF inspection: flag images missing expected camera sensor metadata.

Shadow mode: exceptions caught by orchestration layer.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from orchestration.results import TamperingResult

# ELA re-save quality and anomaly threshold
_ELA_QUALITY = 75
_ELA_SCORE_THRESHOLD = 0.18  # normalised; tune after shadow-mode data accumulates

# EXIF tags expected from a real camera photo
_EXPECTED_EXIF_TAGS = {
    0x010F,  # Make
    0x0110,  # Model
    0x9003,  # DateTimeOriginal
}


def _ela_score(image_bytes: bytes) -> float:
    """Compute normalised ELA residual energy.

    Re-saves the JPEG at _ELA_QUALITY, computes per-pixel absolute diff,
    returns mean diff normalised to 0–1.
    """
    from PIL import Image  # type: ignore

    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=_ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr = np.array(original, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    diff = np.abs(orig_arr - recomp_arr)
    return float(diff.mean() / 255.0)


def _check_exif(image_bytes: bytes) -> bool:
    """Return True (suspicious) if key EXIF camera tags are absent."""
    try:
        from PIL import Image  # type: ignore
        from PIL.ExifTags import TAGS  # type: ignore

        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()  # type: ignore[attr-defined]
        if exif_data is None:
            return True  # no EXIF at all — suspicious
        present = set(exif_data.keys())
        missing = _EXPECTED_EXIF_TAGS - present
        # Flag if more than one expected tag is missing
        return len(missing) >= 2
    except Exception:
        return False  # if EXIF can't be read, don't flag


def run(image_bytes: bytes) -> TamperingResult:
    """Detect digital tampering via ELA and EXIF analysis.

    Args:
        image_bytes: raw JPEG bytes from S3 upload.

    Returns:
        TamperingResult with is_tampered, ela_score, exif_suspicious.
    """
    ela = _ela_score(image_bytes)
    exif_suspicious = _check_exif(image_bytes)

    is_tampered = ela >= _ELA_SCORE_THRESHOLD or exif_suspicious

    return TamperingResult(
        is_tampered=is_tampered,
        ela_score=ela,
        exif_suspicious=exif_suspicious,
        label="tampered" if is_tampered else "clean",
    )
