"""Stage 7 — Orchestration & Shadow Mode Decision Matrix.

Entry point: run(image_bytes) -> PipelineResult

Pipeline order
──────────────
1. quality_gate  — 8-class quality/attack classifier on the raw image.
                   Rejects: screen | printout | selfie | back | garbage → stop.
                   Live:    good_front | partial | blurry → proceed.
2. id_crop       — YOLO corner detection, warp, orientation correction.
3. tampering     — ELA + EXIF analysis.
4. id_type       — 8-class ID type classifier on cropped image.
5. field_extract — AWS Textract OCR (skipped in shadow mode if env var set).

Phase 1 (shadow mode): all stages run asynchronously and write scores only.
They NEVER block or modify the user journey. Every stage call is wrapped in
try/except — a model failure is invisible to the end user.

Risk tiers (post-shadow, Phase 2):
  high_fraud   — ELA tampering or screen replay with high confidence
  garbage_photo — no ID detected, selfie, blank screen, back of card
  user_error   — partial document, heavy rotation
  clean        — passes all checks
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from orchestration.results import (
    ExtractedFields,
    FieldExtractResult,
    IdCropResult,
    IdTypeResult,
    PipelineResult,
    PresentationAttackResult,
    QualityGateResult,
    TamperingResult,
)

logger = logging.getLogger(__name__)

_REJECT_LABELS = {"screen", "printout", "selfie", "back", "garbage"}


def _load_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to BGR numpy array."""
    import cv2  # type: ignore
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _risk_tier(result: PipelineResult) -> str:
    qg_label = result.quality_gate.label if result.quality_gate else None
    if result.is_tampered or (
        result.is_screen_replay
        and result.quality_gate is not None
        and result.quality_gate.confidence > 0.85
    ):
        return "high_fraud"
    if qg_label in {"screen", "printout", "selfie", "back", "garbage"}:
        return "garbage_photo"
    if result.crop and result.crop.label in (
        "no_id_detected",
        "invalid_crop",
    ):
        return "garbage_photo"
    if result.is_partial_document:
        return "user_error"
    return "clean"


def run(image_bytes: bytes) -> PipelineResult:
    """Run all pipeline stages in shadow mode.

    Each stage is wrapped in try/except. Failures are logged but never raised.
    Scores are written to DB by the caller; this function only returns the result.

    Args:
        image_bytes: raw JPEG bytes from S3 upload.

    Returns:
        PipelineResult with all stage outputs and risk_tier.
    """
    result = PipelineResult()

    # ── Decode image once ─────────────────────────────────────────────────────
    try:
        image = _load_image_bytes(image_bytes)
    except Exception as exc:
        logger.error("image decode failed: %s", exc)
        image = np.zeros((1, 1, 3), dtype=np.uint8)

    # ── Quality Gate (runs first on raw image) ────────────────────────────────
    try:
        import quality_gate as qg_module
        qg_result: QualityGateResult = qg_module.run(image)
        result.quality_gate = qg_result
        result.is_screen_replay = qg_result.is_screen_replay
        result.is_printout = qg_result.is_printout
    except Exception as exc:
        logger.error("quality_gate stage failed: %s", exc)
        # Permissive fallback — let image proceed rather than silently block it
        result.quality_gate = QualityGateResult(
            label="good_front", confidence=0.0,
            is_live=True, is_screen_replay=False, is_printout=False,
        )

    # ── Early exit for rejected images ────────────────────────────────────────
    if not result.quality_gate.is_live:
        result.risk_tier = _risk_tier(result)
        result.label = result.risk_tier
        return result

    # ── Stage 1: ID Crop (only for live images) ───────────────────────────────
    cropped_image = image
    try:
        import id_crop
        crop_result: IdCropResult = id_crop.run(image)
        result.crop = crop_result
        result.is_partial_document = crop_result.is_partial_document
        if crop_result.cropped_image is not None:
            cropped_image = crop_result.cropped_image
    except Exception as exc:
        logger.error("id_crop stage failed: %s", exc)

    # ── Stage 3: Tampering Detection (runs on raw bytes) ─────────────────────
    try:
        import tampering_detection
        tamper_result: TamperingResult = tampering_detection.run(image_bytes)
        result.tampering = tamper_result
        result.is_tampered = tamper_result.is_tampered
    except Exception as exc:
        logger.error("tampering_detection stage failed: %s", exc)

    # ── Stage 4: ID Type Classification ──────────────────────────────────────
    try:
        crop_label = result.crop.label if result.crop else None
        from id_crop.quality import crop_ready_for_classification

        ready, skip_reason = crop_ready_for_classification(cropped_image, crop_label)
        if not ready:
            logger.info("id_type skipped: %s", skip_reason)
            result.id_type = IdTypeResult(id_type="unknown_id", confidence=0.0)
            result.id_type_label = "unknown_id"
        else:
            import id_type as id_type_module
            id_type_result: IdTypeResult = id_type_module.run(cropped_image)
            result.id_type = id_type_result
            result.id_type_label = id_type_result.id_type
    except Exception as exc:
        logger.error("id_type stage failed: %s", exc)

    # ── Stage 5: Field Extraction ─────────────────────────────────────────────
    try:
        import field_extractor
        field_result: FieldExtractResult = field_extractor.run(
            image_bytes, id_type=result.id_type_label
        )
        result.field_extract = field_result
        result.extracted_fields = field_result.extracted_fields
        result.field_extraction_confidence = field_result.field_extraction_confidence
    except Exception as exc:
        logger.error("field_extractor stage failed: %s", exc)

    # ── Risk tier ─────────────────────────────────────────────────────────────
    result.risk_tier = _risk_tier(result)
    result.label = result.risk_tier

    return result
