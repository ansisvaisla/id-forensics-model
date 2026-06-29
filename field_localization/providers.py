"""OCR provider adapters for Stage 4 field localization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OCRWord:
    """Single OCR word with normalized 0-1 geometry."""

    text: str
    confidence: float
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


def from_aws_rekognition_response(response_summary_json: dict[str, Any]) -> list[OCRWord]:
    """Convert stored AWS Rekognition DetectText output into OCRWord objects."""
    words: list[OCRWord] = []
    for item in response_summary_json.get("textDetections", []):
        if item.get("type") != "WORD":
            continue
        text = str(item.get("detectedText", "")).strip()
        if not text:
            continue
        box = item.get("geometry", {}).get("boundingBox", {})
        words.append(
            OCRWord(
                text=text,
                confidence=float(item.get("confidence", 0.0)) / 100.0,
                left=float(box.get("left", 0.0)),
                top=float(box.get("top", 0.0)),
                width=float(box.get("width", 0.0)),
                height=float(box.get("height", 0.0)),
            )
        )
    return words


def run_local_ocr(image: np.ndarray) -> list[OCRWord]:
    """Run a local OCR engine if one is installed.

    PaddleOCR is tried first because it tends to perform better on document text.
    EasyOCR is a fallback. If neither is installed, return an empty list so Stage 4
    remains non-blocking in shadow mode.
    """
    words = _run_paddleocr(image)
    if words:
        return words
    return _run_easyocr(image)


def _run_paddleocr(image: np.ndarray) -> list[OCRWord]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError:
        return []

    h, w = image.shape[:2]
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(image, cls=True)
    words: list[OCRWord] = []
    for page in result or []:
        for det in page or []:
            if len(det) != 2:
                continue
            box, payload = det
            if len(payload) != 2:
                continue
            text, conf = payload
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            left, top = max(min(xs) / w, 0.0), max(min(ys) / h, 0.0)
            right, bottom = min(max(xs) / w, 1.0), min(max(ys) / h, 1.0)
            if str(text).strip():
                words.append(
                    OCRWord(str(text).strip(), float(conf), left, top, right - left, bottom - top)
                )
    return words


def _run_easyocr(image: np.ndarray) -> list[OCRWord]:
    try:
        import easyocr  # type: ignore
    except ImportError:
        return []

    h, w = image.shape[:2]
    reader = easyocr.Reader(["en"], gpu=False)
    result = reader.readtext(image)
    words: list[OCRWord] = []
    for box, text, conf in result:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        left, top = max(min(xs) / w, 0.0), max(min(ys) / h, 0.0)
        right, bottom = min(max(xs) / w, 1.0), min(max(ys) / h, 1.0)
        if str(text).strip():
            words.append(
                OCRWord(str(text).strip(), float(conf), left, top, right - left, bottom - top)
            )
    return words
