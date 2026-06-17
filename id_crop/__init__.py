"""Stage 1 — ID Crop.

Entry point: run(image) -> IdCropResult

Detects 4 corners of an ID card using YOLOv8-OBB, applies warpPerspective
to deskew the card, and corrects orientation via a 4-class classifier.

Shadow mode: all exceptions are caught by the orchestration layer.
This module raises normally — the caller handles try/except.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from orchestration.results import IdCropResult

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "stage1_corners" / "weights" / "best.pt"


def _load_model():
    """Lazy-load YOLOv8-OBB model. Raises ImportError if ultralytics not installed."""
    from ultralytics import YOLO  # type: ignore
    return YOLO(str(MODEL_PATH))


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _warp_perspective(image: np.ndarray, corners: list[list[float]]) -> np.ndarray:
    """Deskew image using 4 detected corners in TL, TR, BR, BL order."""
    import cv2  # type: ignore
    src = np.array(corners, dtype=np.float32)
    w = int(max(
        np.linalg.norm(src[1] - src[0]),
        np.linalg.norm(src[2] - src[3]),
    ))
    h = int(max(
        np.linalg.norm(src[3] - src[0]),
        np.linalg.norm(src[2] - src[1]),
    ))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def _correct_orientation(image: np.ndarray) -> np.ndarray:
    """Apply 4-class orientation correction (0/90/180/270 degrees).

    v1: heuristic using image aspect ratio and text alignment.
    v2: replace with dedicated CNN when orientation labels exist.
    """
    import cv2  # type: ignore
    h, w = image.shape[:2]
    # Landscape IDs should be wider than tall; rotate 90 if taller
    if h > w * 1.2:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def run(image: np.ndarray) -> IdCropResult:
    """Detect ID card corners, deskew, and correct orientation.

    Args:
        image: BGR numpy array (OpenCV format).

    Returns:
        IdCropResult with cropped_image set to deskewed card,
        or original image if fewer than 4 corners detected.
    """
    if not MODEL_PATH.is_file():
        # Model not yet trained — return stub result
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="model_not_trained",
            confidence=0.0,
        )

    model = _get_model()
    h, w = image.shape[:2]
    results = model(image, verbose=False)

    if not results or len(results[0].obb) == 0:
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="no_id_detected",
            confidence=0.0,
        )

    # Use highest-confidence detection
    best = results[0].obb[0]
    conf = float(best.conf)
    pts = best.xyxyxyxy[0].cpu().numpy().reshape(4, 2).tolist()

    is_partial = len(pts) < 4
    if is_partial:
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=len(pts),
            label="partial_id",
            confidence=conf,
        )

    cropped = _warp_perspective(image, pts)
    cropped = _correct_orientation(cropped)

    return IdCropResult(
        cropped_image=cropped,
        is_partial_document=False,
        corners_detected=4,
        label="id_card",
        confidence=conf,
    )
