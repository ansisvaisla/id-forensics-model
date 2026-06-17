"""Tests for Stage 3 — Tampering Detection (algorithmic, no model required)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture_bytes(name: str) -> bytes:
    path = FIXTURES / name
    if not path.is_file():
        pytest.skip(f"Fixture not found: {path}")
    return path.read_bytes()


def _make_synthetic_jpeg(width: int = 100, height: int = 60) -> bytes:
    """Create a minimal synthetic JPEG for testing (no real EXIF)."""
    from PIL import Image  # type: ignore
    import numpy as np

    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_ela_score_returns_float() -> None:
    from tampering_detection import run

    jpeg = _make_synthetic_jpeg()
    result = run(jpeg)
    assert isinstance(result.ela_score, float)
    assert 0.0 <= result.ela_score <= 1.0


def test_ela_result_has_required_fields() -> None:
    from tampering_detection import run

    jpeg = _make_synthetic_jpeg()
    result = run(jpeg)
    assert hasattr(result, "is_tampered")
    assert hasattr(result, "ela_score")
    assert hasattr(result, "exif_suspicious")
    assert result.label in ("tampered", "clean", "unknown")


def test_synthetic_image_missing_exif_flagged() -> None:
    """Synthetic PIL image has no EXIF → exif_suspicious should be True."""
    from tampering_detection import run

    jpeg = _make_synthetic_jpeg()
    result = run(jpeg)
    assert result.exif_suspicious is True


def test_invalid_bytes_raises_or_returns_gracefully() -> None:
    """Corrupt bytes should raise an exception (orchestration catches it)."""
    from tampering_detection import run

    with pytest.raises(Exception):
        run(b"not_a_jpeg")
