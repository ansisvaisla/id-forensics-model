"""Geometry helpers for mapping full-image labels to canonical ID coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from field_localization.templates import Zone


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in normalized 0-1 coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)


def order_corners_tl_tr_br_bl(points: list[list[float]] | np.ndarray) -> np.ndarray:
    """Return four normalized points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected four corner points, got shape {pts.shape}")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def homography_full_to_canonical(corners: list[list[float]] | np.ndarray) -> np.ndarray:
    """Build homography from full-image normalized coordinates to canonical ID space."""
    src = order_corners_tl_tr_br_bl(corners)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def homography_canonical_to_full(corners: list[list[float]] | np.ndarray) -> np.ndarray:
    """Build homography from canonical ID space back to full-image normalized coordinates."""
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = order_corners_tl_tr_br_bl(corners)
    return cv2.getPerspectiveTransform(src, dst)


def transform_points(points: list[list[float]] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply perspective transform to normalized points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return transformed


def rect_to_points(rect: Rect | Zone) -> np.ndarray:
    """Return four points for rectangle corners."""
    return np.array(
        [
            [rect.left, rect.top],
            [rect.right, rect.top],
            [rect.right, rect.bottom],
            [rect.left, rect.bottom],
        ],
        dtype=np.float32,
    )


def bbox_from_points(points: list[list[float]] | np.ndarray, clip: bool = True) -> Rect:
    """Convert points to an axis-aligned normalized bounding rectangle."""
    pts = np.asarray(points, dtype=np.float32)
    left = float(np.min(pts[:, 0]))
    top = float(np.min(pts[:, 1]))
    right = float(np.max(pts[:, 0]))
    bottom = float(np.max(pts[:, 1]))
    if clip:
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(1.0, right), min(1.0, bottom)
    return Rect(left=left, top=top, right=right, bottom=bottom)


def transform_rect(rect: Rect | Zone, matrix: np.ndarray, clip: bool = True) -> Rect:
    """Transform a rectangle through a homography and return its axis-aligned bbox."""
    return bbox_from_points(transform_points(rect_to_points(rect), matrix), clip=clip)


def label_studio_rect_to_full(value: dict) -> Rect:
    """Convert Label Studio x/y/width/height percentages to normalized full-image rect."""
    left = float(value.get("x", 0.0)) / 100.0
    top = float(value.get("y", 0.0)) / 100.0
    width = float(value.get("width", 0.0)) / 100.0
    height = float(value.get("height", 0.0)) / 100.0
    return Rect(left=left, top=top, right=left + width, bottom=top + height)


def rect_to_label_studio_value(rect: Rect, label: str, from_name: str = "field") -> dict:
    """Convert normalized rect into Label Studio rectangle value."""
    labels_key = "rectanglelabels"
    return {
        "from_name": from_name,
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": round(rect.left * 100.0, 3),
            "y": round(rect.top * 100.0, 3),
            "width": round(rect.width * 100.0, 3),
            "height": round(rect.height * 100.0, 3),
            labels_key: [label],
        },
    }
