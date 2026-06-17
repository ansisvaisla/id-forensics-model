"""Stage 7 — Orchestration & Shadow Mode Decision Matrix.

Entry point: run(image_bytes) -> PipelineResult

Phase 1 (shadow mode): all stages run asynchronously and write scores only.
They NEVER block or modify the user journey. Every stage call is wrapped in
try/except — a model failure is invisible to the end user.

Risk tiers (post-shadow, Phase 2):
  high_fraud   — ELA tampering or screen replay with high confidence
  garbage_photo — no ID detected, selfie, blank screen
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
    TamperingResult,
)

logger = logging.getLogger(__name__)


def _load_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to BGR numpy array."""
    import cv2  # type: ignore
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _risk_tier(result: PipelineResult) -> str:
    if result.is_tampered or (result.is_screen_replay and (result.presentation_attack and result.presentation_attack.screen_score > 0.85)):
        return "high_fraud"
    if result.crop and result.crop.label in ("no_id_detected", "selfie_instead_of_document"):
        return "garbage_photo"
    if result.is_partial_document:
        return "user_error"
    return "clean"


def run(image_bytes: bytes) -> PipelineResult:
    """Run all pipeline stages in shadow mode.

    Each stage is wrapped in try/except. Failures are logged but never raised.
    Scores are written to DB by the caller; this function only returns the result object.

    Args:
        image_bytes: raw JPEG bytes from S3 upload.

    Returns:
        PipelineResult with all stage outputs and risk_tier.
    """
    result = PipelineResult()

    # --- Stage 1: ID Crop ---
    try:
        image = _load_image_bytes(image_bytes)
        import id_crop
        crop_result: IdCropResult = id_crop.run(image)
        result.crop = crop_result
        result.is_partial_document = crop_result.is_partial_document
        cropped_image = crop_result.cropped_image if crop_result.cropped_image is not None else image
    except Exception as exc:
        logger.error("id_crop stage failed: %s", exc)
        try:
            image = _load_image_bytes(image_bytes)
            cropped_image = image
        except Exception:
            image = np.zeros((1, 1, 3), dtype=np.uint8)
            cropped_image = image

    # --- Stage 2: Presentation Attack (runs on raw upload, not cropped) ---
    try:
        import presentation_attack
        pad_result: PresentationAttackResult = presentation_attack.run(image)
        result.presentation_attack = pad_result
        result.is_screen_replay = pad_result.is_screen_replay
        result.is_printout = pad_result.is_printout
    except Exception as exc:
        logger.error("presentation_attack stage failed: %s", exc)

    # --- Stage 3: Tampering Detection (runs on raw bytes) ---
    try:
        import tampering_detection
        tamper_result: TamperingResult = tampering_detection.run(image_bytes)
        result.tampering = tamper_result
        result.is_tampered = tamper_result.is_tampered
    except Exception as exc:
        logger.error("tampering_detection stage failed: %s", exc)

    # --- Stage 4: ID Type Classification ---
    try:
        import id_type as id_type_module
        id_type_result: IdTypeResult = id_type_module.run(cropped_image)
        result.id_type = id_type_result
        result.id_type_label = id_type_result.id_type
    except Exception as exc:
        logger.error("id_type stage failed: %s", exc)

    # --- Stage 5: Field Extraction ---
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

    # --- Stage 7: Risk tier ---
    result.risk_tier = _risk_tier(result)
    result.label = result.risk_tier

    return result
