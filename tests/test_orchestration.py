"""Tests for orchestration shadow mode (no real models needed)."""
from __future__ import annotations

import io

import pytest


def _make_synthetic_jpeg() -> bytes:
    from PIL import Image  # type: ignore
    import numpy as np

    arr = np.random.randint(0, 200, (60, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_orchestration_returns_pipeline_result() -> None:
    """Orchestration must return PipelineResult even without trained models."""
    from orchestration import run
    from orchestration.results import PipelineResult

    jpeg = _make_synthetic_jpeg()
    result = run(jpeg)
    assert isinstance(result, PipelineResult)


def test_orchestration_never_raises() -> None:
    """Shadow mode contract: orchestration must never raise even on bad input."""
    from orchestration import run

    # Empty bytes — all stages should fail gracefully
    try:
        result = run(b"")
        # If it returns, it should be a PipelineResult
        from orchestration.results import PipelineResult
        assert isinstance(result, PipelineResult)
    except Exception as exc:
        pytest.fail(f"Orchestration raised an exception in shadow mode: {exc}")


def test_pipeline_result_has_metadata_flags() -> None:
    from orchestration import run

    jpeg = _make_synthetic_jpeg()
    result = run(jpeg)
    assert hasattr(result, "is_partial_document")
    assert hasattr(result, "is_screen_replay")
    assert hasattr(result, "is_printout")
    assert hasattr(result, "is_tampered")
    assert hasattr(result, "risk_tier")
    assert hasattr(result, "extracted_fields")
