"""Stage 1 — ID Crop.

Entry point: run(image) -> IdCropResult

Detects 4 corners of an ID card using YOLOv8-Pose (keypoint detection),
applies warpPerspective to deskew the card, and corrects orientation.

Using Pose instead of OBB means the model predicts 4 arbitrary keypoints
rather than a rotated rectangle — correctly handling perspective-skewed
(trapezoidal) cards.

After keypoint prediction, an optional edge-detection refinement step snaps
corners to actual card edges when the image has sufficient contrast.

Shadow mode: all exceptions are caught by the orchestration layer.
This module raises normally — the caller handles try/except.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from orchestration.results import IdCropResult

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "stage1_corners" / "weights" / "best.pt"

_model = None


def _load_model():
    from ultralytics import YOLO  # type: ignore
    return YOLO(str(MODEL_PATH))


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _extract_corners(result) -> Optional[np.ndarray]:
    """Extract (4, 2) corner array from a YOLO result.

    Supports both Pose (keypoints) and OBB models so old weights still work
    during transition. Returns None if no detection.
    """
    # Pose model (preferred)
    if (
        hasattr(result, "keypoints")
        and result.keypoints is not None
        and len(result.keypoints) > 0
    ):
        kpts = result.keypoints[0]
        if kpts.xy is not None and len(kpts.xy) > 0:
            pts = kpts.xy[0].cpu().numpy()  # (4, 2) pixel coords
            if len(pts) == 4:
                return pts.astype(np.float32)

    # OBB fallback (legacy weights)
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        pts = result.obb[0].xyxyxyxy[0].cpu().numpy().reshape(4, 2)
        return pts.astype(np.float32)

    return None


def _get_conf(result) -> float:
    """Extract detection confidence from either Pose or OBB result."""
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        return float(result.boxes[0].conf)
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        return float(result.obb[0].conf)
    return 0.0


def _refine_corners_edge(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Snap predicted corners to actual card edges via Canny contour detection.

    Uses the predicted corners as a region of interest, finds the largest
    quadrilateral contour within it, and returns refined corners if they are
    plausible (within 10% of image diagonal from the predictions).
    Falls back to the original corners if refinement is unreliable.
    """
    import cv2
    import math
    from itertools import permutations

    h, w = image.shape[:2]
    diag = math.hypot(w, h)

    # Padded bounding box around predicted corners
    PAD = int(0.04 * min(w, h))
    x1 = max(0, int(corners[:, 0].min()) - PAD)
    y1 = max(0, int(corners[:, 1].min()) - PAD)
    x2 = min(w, int(corners[:, 0].max()) + PAD)
    y2 = min(h, int(corners[:, 1].max()) + PAD)

    region = image[y1:y2, x1:x2]
    if region.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
        return corners

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Auto-threshold with Otsu then Canny
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, thresh * 0.5, thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return corners

    # Try contours from largest to smallest, accept first 4-point polygon
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Map back to full-image coordinates
        refined = approx.reshape(4, 2).astype(np.float32)
        refined[:, 0] += x1
        refined[:, 1] += y1

        # Accept only if refined corners are close to predictions
        best_dist = min(
            sum(math.hypot(refined[perm[i], 0] - corners[i, 0],
                           refined[perm[i], 1] - corners[i, 1])
                for i in range(4)) / 4
            for perm in permutations(range(4))
        )
        if best_dist < 0.10 * diag:
            return refined

    return corners


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corners into TL, TR, BR, BL order for warpPerspective."""
    # Sum: TL has smallest sum, BR has largest
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Deskew image using 4 detected corners ordered TL, TR, BR, BL."""
    import cv2  # type: ignore
    src = _order_corners_tl_tr_br_bl(corners)
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
    """Rotate portrait-oriented crops to landscape (ID cards are always landscape)."""
    import cv2  # type: ignore
    h, w = image.shape[:2]
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
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="model_not_trained",
            confidence=0.0,
        )

    model = _get_model()
    results = model(image, verbose=False)

    if not results:
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="no_id_detected",
            confidence=0.0,
        )

    corners = _extract_corners(results[0])
    conf = _get_conf(results[0])

    if corners is None or len(corners) < 4:
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="no_id_detected",
            confidence=conf,
        )

    # Refine corners using edge detection
    corners = _refine_corners_edge(image, corners)

    cropped = _warp_perspective(image, corners)
    cropped = _correct_orientation(cropped)

    return IdCropResult(
        cropped_image=cropped,
        is_partial_document=False,
        corners_detected=4,
        label="id_card",
        confidence=conf,
    )
