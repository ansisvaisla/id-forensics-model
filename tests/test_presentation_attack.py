"""Tests for deprecated presentation_attack compatibility module."""
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


def _jpeg_to_bgr(jpeg: bytes):
    import cv2  # type: ignore
    import numpy as np

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def test_stub_returns_result_when_model_missing() -> None:
    """Without a trained model, presentation_attack must return a valid stub result."""
    from presentation_attack import run
    from orchestration.results import QualityGateResult

    jpeg = _make_synthetic_jpeg()
    image = _jpeg_to_bgr(jpeg)
    result = run(image)
    assert isinstance(result, QualityGateResult)
    assert result.label == "model_not_trained"
    assert result.is_live is True
    assert result.is_screen_replay is False
    assert result.is_printout is False
