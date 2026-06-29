"""Stage 4 — Field Localization And OCR.

Entry point: run(image, id_type, ocr_words=None) -> FieldExtractResult

The stage is layout-aware:
  1. Get OCR words from an injected provider output or a local OCR engine.
  2. Select a normalized template for the Stage 3 ID type.
  3. Assign OCR words to field zones.
  4. Parse structured values from each zone.

Shadow mode: if no OCR engine is installed or OCR returns no words, the stage
returns a skipped/failed result rather than blocking the pipeline.
"""
from __future__ import annotations

import os

import numpy as np

from orchestration.results import ExtractedFields, FieldExtractResult

from field_localization.parsers import avg_confidence, parse_fields
from field_localization.providers import OCRWord, run_local_ocr
from field_localization.templates import assign_words_to_fields, get_template


def run(
    image: np.ndarray,
    id_type: str = "unknown",
    ocr_words: list[OCRWord] | None = None,
) -> FieldExtractResult:
    """Extract structured ID fields from a cropped/warped card image."""
    if os.getenv("SKIP_FIELD_EXTRACTOR") and ocr_words is None:
        return FieldExtractResult(
            extracted_fields=ExtractedFields(),
            field_extraction_confidence=0.0,
            label="skipped",
        )

    words = ocr_words if ocr_words is not None else run_local_ocr(image)
    if not words:
        return FieldExtractResult(
            extracted_fields=ExtractedFields(),
            field_extraction_confidence=0.0,
            label="failed",
        )

    template = get_template(id_type)
    grouped = assign_words_to_fields(words, template)
    fields = parse_fields(id_type, grouped)
    confidence = avg_confidence(words)
    filled = sum(
        1
        for value in (
            fields.name,
            fields.surname,
            fields.id_number,
            fields.date_of_birth,
            fields.sex,
        )
        if value
    )
    label = "extracted" if filled >= 3 else ("partial" if filled >= 1 else "failed")
    return FieldExtractResult(
        extracted_fields=fields,
        field_extraction_confidence=confidence,
        label=label,
    )


__all__ = ["OCRWord", "run"]
