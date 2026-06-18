"""Stage 1 — ID Crop.

Entry point: run(image) -> IdCropResult

Detects 4 corners of an ID card using YOLOv8-Pose (keypoint detection),
applies warpPerspective to deskew the card, and corrects orientation.

Corner handling strategy (in order):
  1. Confidence gate   — if box or keypoint confidence too low → skip warp
  2. Coverage check    — if card fills >85% of frame → already cropped, skip warp
  3. Off-screen check  — if any corner near image border → partial, skip warp
  4. 3-corner recovery — if exactly 1 corner weak → reconstruct from ID-1 aspect ratio
  5. Edge refinement   — snap corners to actual card edges via Canny
  6. Warp + orient     — perspective transform + landscape correction

Shadow mode: all exceptions are caught by the orchestration layer.
This module raises normally — the caller handles try/except.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from orchestration.results import IdCropResult

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "stage1_corners" / "weights" / "best.pt"

# ISO/IEC 7810 ID-1 card dimensions (mm) — all Kenyan IDs, driving licences
_ID1_ASPECT = 85.6 / 54.0  # ~1.585  (width / height)

# Confidence gates
_MIN_KPT_CONF = 0.25   # minimum per-keypoint confidence to trust that corner
_MIN_KPT_MEAN = 0.40   # minimum mean keypoint confidence across all 4
_MIN_BOX_CONF = 0.45   # minimum bounding box confidence

# Skip warp if card already fills most of frame (phone crops close)
_COVERAGE_SKIP = 0.85  # bbox_area / image_area > this → already well-framed

# Mark as off-screen if corner is within this fraction of image edge
_EDGE_MARGIN = 0.03   # 3% of image dimension

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
    if (
        hasattr(result, "keypoints")
        and result.keypoints is not None
        and len(result.keypoints) > 0
    ):
        kpts = result.keypoints[0]
        if kpts.xy is not None and len(kpts.xy) > 0:
            pts = kpts.xy[0].cpu().numpy()
            if len(pts) == 4:
                return pts.astype(np.float32)

    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        pts = result.obb[0].xyxyxyxy[0].cpu().numpy().reshape(4, 2)
        return pts.astype(np.float32)

    return None


def _get_kpt_confidences(result) -> Optional[np.ndarray]:
    """Per-keypoint confidence array (length 4) from a Pose result."""
    if (
        hasattr(result, "keypoints")
        and result.keypoints is not None
        and len(result.keypoints) > 0
    ):
        kpts = result.keypoints[0]
        if kpts.conf is not None and len(kpts.conf) > 0:
            return kpts.conf[0].cpu().numpy()
    return None


def _get_conf(result) -> float:
    """Extract detection confidence from either Pose or OBB result."""
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        return float(result.boxes[0].conf)
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        return float(result.obb[0].conf)
    return 0.0


def _corners_reliable(box_conf: float, kpt_conf: Optional[np.ndarray]) -> bool:
    """Return False when corners are too uncertain to deskew safely."""
    if box_conf < _MIN_BOX_CONF:
        return False
    if kpt_conf is None or len(kpt_conf) < 4:
        return True  # OBB fallback — no per-keypoint scores
    return float(kpt_conf.min()) >= _MIN_KPT_CONF and float(kpt_conf.mean()) >= _MIN_KPT_MEAN


def _card_fills_frame(corners: np.ndarray, img_w: int, img_h: int) -> bool:
    """True if predicted card bbox covers most of the image (already well-framed)."""
    x1, y1 = corners[:, 0].min(), corners[:, 1].min()
    x2, y2 = corners[:, 0].max(), corners[:, 1].max()
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    return (bbox_area / img_area) > _COVERAGE_SKIP


def _offscreen_mask(corners: np.ndarray, img_w: int, img_h: int) -> list[bool]:
    """Return bool list: True = that corner is near/outside the image border."""
    mx = _EDGE_MARGIN * img_w
    my = _EDGE_MARGIN * img_h
    mask = []
    for x, y in corners:
        mask.append(
            x < mx or x > img_w - mx or y < my or y > img_h - my
        )
    return mask


def _reconstruct_missing_corner(
    corners: np.ndarray,
    bad_idx: int,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Reconstruct the missing corner using ID-1 aspect ratio.

    Strategy: find the 3 good corners, project the 4th using the known
    card aspect ratio (1.585:1). Works when exactly one corner is off-screen.

    The 4 corners are in arbitrary order; we first sort to TL/TR/BR/BL,
    reconstruct the missing one, then return in original order.
    """
    ordered = _order_corners_tl_tr_br_bl(corners)
    bad_ordered = _find_ordered_idx(corners, ordered, bad_idx)

    # TL=0, TR=1, BR=2, BL=3
    tl, tr, br, bl = ordered[0], ordered[1], ordered[2], ordered[3]

    if bad_ordered == 0:  # missing TL
        # TL ≈ TR - (BR - BL) + (BL - BR)*0 … simpler: use parallelogram
        tl = tr + bl - br
        ordered[0] = tl
    elif bad_ordered == 1:  # missing TR
        tr = tl + br - bl
        ordered[1] = tr
    elif bad_ordered == 2:  # missing BR
        br = bl + tr - tl
        ordered[2] = br
    else:  # missing BL
        bl = br + tl - tr
        ordered[3] = bl

    # Clip to image bounds with small margin
    ordered[:, 0] = np.clip(ordered[:, 0], 0, img_w - 1)
    ordered[:, 1] = np.clip(ordered[:, 1], 0, img_h - 1)

    # Map back to original corner order
    result = corners.copy()
    for orig_i, ordered_i in enumerate(_ordered_to_orig_map(corners, ordered)):
        result[orig_i] = ordered[ordered_i]
    return result


def _find_ordered_idx(orig: np.ndarray, ordered: np.ndarray, orig_idx: int) -> int:
    """Find which TL/TR/BR/BL index corresponds to orig[orig_idx]."""
    pt = orig[orig_idx]
    dists = [math.hypot(float(ordered[i, 0] - pt[0]), float(ordered[i, 1] - pt[1]))
             for i in range(4)]
    return int(np.argmin(dists))


def _ordered_to_orig_map(orig: np.ndarray, ordered: np.ndarray) -> list[int]:
    """For each original corner index, return which ordered index it maps to."""
    mapping = []
    for pt in orig:
        dists = [math.hypot(float(ordered[i, 0] - pt[0]), float(ordered[i, 1] - pt[1]))
                 for i in range(4)]
        mapping.append(int(np.argmin(dists)))
    return mapping


def _refine_corners_edge(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Snap predicted corners to actual card edges via Canny contour detection."""
    import cv2
    from itertools import permutations

    h, w = image.shape[:2]
    diag = math.hypot(w, h)

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
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(otsu_val) if float(otsu_val) > 0 else 50.0
    edges = cv2.Canny(blurred, t * 0.5, t)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return corners

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        refined = approx.reshape(4, 2).astype(np.float32)
        refined[:, 0] += x1
        refined[:, 1] += y1
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
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Deskew image using 4 detected corners ordered TL, TR, BR, BL."""
    import cv2
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
    import cv2
    h, w = image.shape[:2]
    if h > w * 1.2:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def run(image: np.ndarray) -> IdCropResult:
    """Detect ID card corners, deskew, and correct orientation.

    Returns IdCropResult with cropped_image set to deskewed card,
    or original image if cropping is not safe/needed.
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
    h, w = image.shape[:2]
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
    kpt_conf = _get_kpt_confidences(results[0])

    if corners is None or len(corners) < 4:
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="no_id_detected",
            confidence=conf,
        )

    # Gate 1: overall confidence too low
    if not _corners_reliable(conf, kpt_conf):
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=4,
            label="low_corner_confidence",
            confidence=conf,
        )

    # Gate 2: card already fills the frame — warp would just add distortion
    if _card_fills_frame(corners, w, h):
        return IdCropResult(
            cropped_image=image,
            is_partial_document=False,
            corners_detected=4,
            label="full_frame_id",
            confidence=conf,
        )

    # Gate 3: partial card — check which corners are off-screen
    offscreen = _offscreen_mask(corners, w, h)
    n_offscreen = sum(offscreen)

    if n_offscreen >= 2:
        # Too many corners missing — can't reliably reconstruct
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=4 - n_offscreen,
            label="partial_id",
            confidence=conf,
        )

    if n_offscreen == 1:
        # Exactly one corner off-screen — reconstruct using parallelogram
        bad_idx = offscreen.index(True)
        corners = _reconstruct_missing_corner(corners, bad_idx, w, h)

    # Refine + warp
    corners = _refine_corners_edge(image, corners)
    cropped = _warp_perspective(image, corners)
    cropped = _correct_orientation(cropped)

    is_partial = n_offscreen > 0
    return IdCropResult(
        cropped_image=cropped,
        is_partial_document=is_partial,
        corners_detected=4,
        label="id_card_reconstructed" if is_partial else "id_card",
        confidence=conf,
    )
