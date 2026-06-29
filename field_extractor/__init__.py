"""Compatibility wrapper for Stage 4 — Field Localization And OCR.

New code should import field_localization directly. This module keeps the old
run(image_bytes, id_type) API alive for scripts/tests that still call it.
"""
from __future__ import annotations

import os
import numpy as np

from orchestration.results import ExtractedFields, FieldExtractResult


def run(image_bytes: bytes, id_type: str = "unknown") -> FieldExtractResult:
    """Decode bytes and delegate to field_localization.run()."""
    if os.getenv("SKIP_FIELD_EXTRACTOR"):
        return FieldExtractResult(
            extracted_fields=ExtractedFields(),
            field_extraction_confidence=0.0,
            label="skipped",
        )
    import cv2  # type: ignore
    import field_localization

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return FieldExtractResult(
            extracted_fields=ExtractedFields(),
            field_extraction_confidence=0.0,
            label="failed",
        )
    return field_localization.run(image, id_type=id_type)
