"""Stage 1 — ID Crop.

Entry point: run(image) -> IdCropResult

Detection strategy:
  1. ML bounding-box crop (primary for Stage 4) — detects where the card is,
     crops the bbox region. Reliable, never produces face crops or strip artifacts.
  2. Perspective warp (attempted for Stage 5 OCR quality) — only if ML finds
     confident corners AND the quad+warp pass sanity checks.
  3. Full-frame fallback — if card fills >85% of frame, return as-is.

The classical contour approach was removed because it confuses inner card
rectangles (photo region, text boxes) with the card boundary, producing
face crops instead of card crops.

Shadow mode: this module raises normally. The orchestration layer wraps it in try/except.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from orchestration.results import IdCropResult
from id_crop.quality import crop_is_plausible, quad_is_sane

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "stage1_corners" / "weights" / "best.pt"

_ID1_ASPECT = 85.6 / 54.0  # ISO/IEC 7810 ID-1 landscape aspect ratio (~1.585)

# ML fallback confidence gates
_MIN_KPT_CONF = 0.25
_MIN_KPT_MEAN = 0.40
_MIN_BOX_CONF = 0.45

# Skip warp if card already fills most of frame
_COVERAGE_SKIP = 0.85

# Mark as off-screen if corner is within this fraction of image edge
_EDGE_MARGIN = 0.03

_ml_model = None


# ── ML fallback ───────────────────────────────────────────────────────────────

def _load_ml_model():
    from ultralytics import YOLO  # type: ignore
    return YOLO(str(MODEL_PATH))


def _get_ml_model():
    global _ml_model
    if _ml_model is None:
        _ml_model = _load_ml_model()
    return _ml_model


def _ml_detect(image: np.ndarray) -> tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
    """Run YOLOv8 Pose model. Returns (corners, box_conf, kpt_conf) or (None, 0, None)."""
    if not MODEL_PATH.is_file():
        return None, 0.0, None

    model = _get_ml_model()
    results = model(image, verbose=False)
    if not results:
        return None, 0.0, None

    result = results[0]

    # Pose model (keypoints)
    if (
        hasattr(result, "keypoints")
        and result.keypoints is not None
        and len(result.keypoints) > 0
    ):
        kpts = result.keypoints[0]
        if kpts.xy is not None and len(kpts.xy) > 0:
            pts = kpts.xy[0].cpu().numpy()
            if len(pts) == 4:
                conf = 0.0
                if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                    conf = float(result.boxes[0].conf)
                kpt_conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else None
                return pts.astype(np.float32), conf, kpt_conf

    # OBB legacy fallback
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        pts = result.obb[0].xyxyxyxy[0].cpu().numpy().reshape(4, 2)
        conf = float(result.obb[0].conf)
        return pts.astype(np.float32), conf, None

    return None, 0.0, None


def _ml_corners_reliable(box_conf: float, kpt_conf: Optional[np.ndarray]) -> bool:
    if box_conf < _MIN_BOX_CONF:
        return False
    if kpt_conf is None or len(kpt_conf) < 4:
        return True
    return float(kpt_conf.min()) >= _MIN_KPT_CONF and float(kpt_conf.mean()) >= _MIN_KPT_MEAN


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _card_fills_frame(corners: np.ndarray, img_w: int, img_h: int) -> bool:
    x1, y1 = corners[:, 0].min(), corners[:, 1].min()
    x2, y2 = corners[:, 0].max(), corners[:, 1].max()
    return ((x2 - x1) * (y2 - y1)) / (img_w * img_h) > _COVERAGE_SKIP


def _offscreen_mask(corners: np.ndarray, img_w: int, img_h: int) -> list[bool]:
    mx, my = _EDGE_MARGIN * img_w, _EDGE_MARGIN * img_h
    return [x < mx or x > img_w - mx or y < my or y > img_h - my
            for x, y in corners]


def _reconstruct_missing_corner(corners: np.ndarray, bad_idx: int,
                                 img_w: int, img_h: int) -> np.ndarray:
    ordered = _order_corners_tl_tr_br_bl(corners)
    # Map bad original index to ordered index
    pt = corners[bad_idx]
    dists = [math.hypot(float(ordered[i, 0] - pt[0]), float(ordered[i, 1] - pt[1]))
             for i in range(4)]
    bad_ordered = int(np.argmin(dists))

    tl, tr, br, bl = ordered[0], ordered[1], ordered[2], ordered[3]
    if bad_ordered == 0:
        ordered[0] = tr + bl - br
    elif bad_ordered == 1:
        ordered[1] = tl + br - bl
    elif bad_ordered == 2:
        ordered[2] = bl + tr - tl
    else:
        ordered[3] = br + tl - tr

    ordered[:, 0] = np.clip(ordered[:, 0], 0, img_w - 1)
    ordered[:, 1] = np.clip(ordered[:, 1], 0, img_h - 1)

    # Rebuild original-order array from updated ordered
    result = corners.copy()
    for orig_i, pt in enumerate(corners):
        dists2 = [math.hypot(float(ordered[i, 0] - pt[0]), float(ordered[i, 1] - pt[1]))
                  for i in range(4)]
        result[orig_i] = ordered[int(np.argmin(dists2))]
    return result


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def _warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    import cv2
    src = _order_corners_tl_tr_br_bl(corners)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def _correct_orientation(image: np.ndarray) -> np.ndarray:
    import cv2
    h, w = image.shape[:2]
    if h > w * 1.2:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _try_warp(image: np.ndarray, corners: np.ndarray,
              conf: float, is_partial: bool,
              method: str) -> Optional[IdCropResult]:
    """Validate corners, warp, validate crop. Returns IdCropResult or None on failure."""
    sane, _ = quad_is_sane(corners)
    if not sane:
        return None

    cropped = _warp_perspective(image, corners)
    cropped = _correct_orientation(cropped)

    ok, _ = crop_is_plausible(cropped)
    if not ok:
        return None

    return IdCropResult(
        cropped_image=cropped,
        is_partial_document=is_partial,
        corners_detected=4,
        label="id_card_reconstructed" if is_partial else "id_card",
        confidence=conf,
    )


# ── Main entry point ─────────────────────────────────────────────────────────

def _ml_bbox_crop(image: np.ndarray, conf: float, corners: np.ndarray) -> Optional[np.ndarray]:
    """Tight bounding-box crop from ML detection. Adds 5% padding."""
    import cv2
    h, w = image.shape[:2]
    pad_x = (corners[:, 0].max() - corners[:, 0].min()) * 0.05
    pad_y = (corners[:, 1].max() - corners[:, 1].min()) * 0.05
    x1 = max(0, int(corners[:, 0].min() - pad_x))
    y1 = max(0, int(corners[:, 1].min() - pad_y))
    x2 = min(w, int(corners[:, 0].max() + pad_x))
    y2 = min(h, int(corners[:, 1].max() + pad_y))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def run(image: np.ndarray) -> IdCropResult:
    """Detect ID card and crop to card region.

    Primary output: ML bounding-box crop (reliable, no warp artifacts).
    If ML corners are confident and quad is sane, also attempts perspective
    warp — if warp passes quality check, returns the warp (better for OCR).
    Falls back to bbox crop or original image if either approach fails.
    """
    h, w = image.shape[:2]

    if not MODEL_PATH.is_file():
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="model_not_trained",
            confidence=0.0,
        )

    ml_corners, conf, kpt_conf = _ml_detect(image)

    if ml_corners is None:
        return IdCropResult(
            cropped_image=image,
            is_partial_document=True,
            corners_detected=0,
            label="no_id_detected",
            confidence=conf,
        )

    # Coverage check — card fills the frame, no crop needed
    if _card_fills_frame(ml_corners, w, h):
        ok, _ = crop_is_plausible(image)
        return IdCropResult(
            cropped_image=image,
            is_partial_document=False,
            corners_detected=4,
            label="full_frame_id" if ok else "invalid_crop",
            confidence=conf,
        )

    # ── Attempt perspective warp if corners are trustworthy ───────────────────
    if _ml_corners_reliable(conf, kpt_conf):
        offscreen = _offscreen_mask(ml_corners, w, h)
        n_off = sum(offscreen)

        corners_for_warp = ml_corners
        if n_off == 1:
            bad_idx = offscreen.index(True)
            corners_for_warp = _reconstruct_missing_corner(ml_corners, bad_idx, w, h)

        if n_off <= 1:
            warp_result = _try_warp(
                image, corners_for_warp,
                conf=conf, is_partial=(n_off > 0),
                method="ml",
            )
            if warp_result is not None:
                return warp_result

    # ── Bbox crop fallback ────────────────────────────────────────────────────
    bbox_crop = _ml_bbox_crop(image, conf, ml_corners)
    if bbox_crop is not None:
        ok, _ = crop_is_plausible(bbox_crop)
        if ok:
            offscreen = _offscreen_mask(ml_corners, w, h)
            is_partial = sum(offscreen) > 0
            return IdCropResult(
                cropped_image=bbox_crop,
                is_partial_document=is_partial,
                corners_detected=4,
                label="bbox_crop",
                confidence=conf,
            )

    # ── Nothing worked ────────────────────────────────────────────────────────
    return IdCropResult(
        cropped_image=image,
        is_partial_document=True,
        corners_detected=4,
        label="invalid_crop",
        confidence=conf,
    )
