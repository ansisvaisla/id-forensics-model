"""Stage 1 — Quality Gate.

8-class EfficientNet-B0 classifier:
  0  screen       — screen replay attack (phone/monitor)
  1  printout     — printed paper attack
  2  selfie       — selfie submitted instead of ID
  3  back         — back side of ID card
  4  garbage      — blank, black screen, unrecognisable
  5  good_front   — clear, well-lit front of ID  ─┐
  6  partial      — card partially out of frame    ├─ live → proceed to id_crop
  7  blurry       — readable but out of focus     ─┘

Entry point: run(image) -> QualityGateResult

Shadow mode: exceptions are caught by the orchestration layer.
Falls back to is_live=True (permissive) when the model file is absent so
inference keeps working without trained weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestration.results import QualityGateResult

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "stage1_quality_gate" / "best.pt"
)
LEGACY_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "stage2_screen" / "best.pt"
)

CLASS_NAMES = (
    "screen",
    "printout",
    "selfie",
    "back",
    "garbage",
    "good_front",
    "partial",
    "blurry",
)
NUM_CLASSES = len(CLASS_NAMES)

_LIVE_LABELS = {"good_front", "partial", "blurry"}
_THRESHOLD = 0.50  # minimum probability to accept the top class; else fallback to live

_model = None


def _load_model():
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torchvision.models import efficientnet_b0  # type: ignore

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    state = torch.load(str(_model_path()), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _model_path() -> Path:
    """Prefer new Stage 1 path, but keep old Stage 2 artifacts usable."""
    return MODEL_PATH if MODEL_PATH.is_file() else LEGACY_MODEL_PATH


def _preprocess(image: np.ndarray) -> "torch.Tensor":
    import cv2  # type: ignore
    import torchvision.transforms as T  # type: ignore

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(rgb).unsqueeze(0)


def run(image: np.ndarray) -> QualityGateResult:
    """Classify raw upload image into one of 8 quality/attack categories.

    Args:
        image: BGR numpy array — the raw upload before any cropping.

    Returns:
        QualityGateResult. is_live=True means the image should proceed to id_crop.
        Returns is_live=True (permissive fallback) when the model is not yet trained.
    """
    if not _model_path().is_file():
        return QualityGateResult(
            label="good_front",
            confidence=0.0,
            is_live=True,
            is_screen_replay=False,
            is_printout=False,
        )

    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    model = _get_model()
    tensor = _preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    idx = int(probs.argmax())
    confidence = float(probs[idx])
    label = CLASS_NAMES[idx]

    # Low-confidence predictions default to live so good images aren't blocked
    if confidence < _THRESHOLD:
        label = "good_front"
        confidence = float(probs[CLASS_NAMES.index("good_front")])

    is_live = label in _LIVE_LABELS
    return QualityGateResult(
        label=label,
        confidence=confidence,
        is_live=is_live,
        is_screen_replay=(label == "screen"),
        is_printout=(label == "printout"),
    )
