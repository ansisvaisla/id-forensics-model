"""Crop quality checks for Stage 2 — used at inference, not just training.

A bad perspective warp can look like a thin colour strip or flat grey block.
Stage 3 will happily call that 'legacy' at 96% confidence. These checks run
after corner detection and after warp so production never passes garbage downstream.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ISO/IEC 7810 ID-1 (mm): 85.6 x 54 -> landscape aspect ~1.585
_ID1_ASPECT = 85.6 / 54.0
_MIN_WARP_DIM = 80          # px; smaller output is almost always a bad warp
_MIN_FILL_RATIO = 0.25      # min(w,h)/max(w,h); rejects thin strips
_MIN_ASPECT = 1.05          # card must be landscape after orientation fix
_MAX_ASPECT = 2.2
_MIN_GRAY_STD = 15.0        # flat solid-colour blocks
_MIN_LAPLACIAN_VAR = 25.0   # blur / empty crops


def estimate_warp_size(corners: np.ndarray) -> tuple[float, float]:
    """Width and height the warp would produce (before orientation correction)."""
    s = corners.astype(np.float32)
    # Sum/diff ordering — same logic as warp, without full sort dependency
    sums = s.sum(axis=1)
    diffs = np.diff(s, axis=1).flatten()
    tl, br = s[np.argmin(sums)], s[np.argmax(sums)]
    tr, bl = s[np.argmin(diffs)], s[np.argmax(diffs)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    w = float(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])))
    h = float(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))
    return w, h


def quad_is_sane(corners: np.ndarray) -> tuple[bool, str]:
    """Reject corner sets that would produce a degenerate warp."""
    if corners is None or len(corners) != 4:
        return False, "wrong_corner_count"

    w, h = estimate_warp_size(corners)
    if w < _MIN_WARP_DIM or h < _MIN_WARP_DIM:
        return False, "warp_too_small"

    fill = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    if fill < _MIN_FILL_RATIO:
        return False, "warp_thin_strip"

    aspect = w / h if h > 0 else 0.0
    if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
        return False, "warp_bad_aspect"

    # Shoelace area — collapsed or twisted quads have near-zero area
    pts = corners.astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 500.0:
        return False, "quad_collapsed"

    return True, "ok"


def crop_is_plausible(image: np.ndarray) -> tuple[bool, str]:
    """Reject warped images that do not look like a real ID card crop."""
    if image is None or image.size == 0:
        return False, "empty_image"

    h, w = image.shape[:2]
    if w < _MIN_WARP_DIM or h < _MIN_WARP_DIM:
        return False, "crop_too_small"

    fill = min(w, h) / max(w, h)
    if fill < _MIN_FILL_RATIO:
        return False, "crop_thin_strip"

    aspect = w / h if h > 0 else 0.0
    if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
        return False, "crop_bad_aspect"

    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if std < _MIN_GRAY_STD and lap_var < _MIN_LAPLACIAN_VAR:
        return False, "crop_flat_blank"

    if lap_var < _MIN_LAPLACIAN_VAR * 0.5:
        return False, "crop_no_detail"

    return True, "ok"


def crop_ready_for_classification(
    image: np.ndarray,
    crop_label: Optional[str] = None,
) -> tuple[bool, str]:
    """Gate for Stage 3 — only classify when Stage 2 succeeded and crop looks real."""
    success_labels = {"id_card", "id_card_reconstructed", "full_frame_id", "bbox_crop"}
    if crop_label is not None and crop_label not in success_labels:
        return False, f"crop_stage_failed:{crop_label}"

    return crop_is_plausible(image)
