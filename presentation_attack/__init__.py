"""Stage 2 — Presentation Attack Detection.

Entry point: run(image) -> PresentationAttackResult

3-class EfficientNet-B0: screen (0) / printout (1) / live (2).

Shadow mode: exceptions are caught by orchestration layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestration.results import PresentationAttackResult

SCREEN_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "stage2_screen" / "best.pt"
)
_NUM_CLASSES = 3
# Thresholds: flag attack if probability exceeds these values
_SCREEN_THRESHOLD = 0.50
_PRINTOUT_THRESHOLD = 0.50

_screen_model = None


def _load_screen_model():
    """Lazy-load Stage 2 classifier. Raises if model file missing."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torchvision.models import efficientnet_b0  # type: ignore

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, _NUM_CLASSES)
    state = torch.load(str(SCREEN_MODEL_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_screen_model():
    global _screen_model
    if _screen_model is None:
        _screen_model = _load_screen_model()
    return _screen_model


def _preprocess(image: np.ndarray) -> "torch.Tensor":
    """Resize and normalise image for EfficientNet-B0."""
    import cv2  # type: ignore
    import torch  # type: ignore
    import torchvision.transforms as T  # type: ignore

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(rgb).unsqueeze(0)


def run(image: np.ndarray) -> PresentationAttackResult:
    """Classify image as screen replay, printout, or live genuine ID.

    Args:
        image: BGR numpy array (cropped or full — runs on raw upload in shadow mode).

    Returns:
        PresentationAttackResult with is_screen_replay, is_printout, and per-class scores.
        Classes: 0=screen, 1=printout, 2=live.
    """
    if not SCREEN_MODEL_PATH.is_file():
        return PresentationAttackResult(
            is_screen_replay=False,
            is_printout=False,
            screen_score=0.0,
            printout_score=0.0,
            label="model_not_trained",
        )

    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    model = _get_screen_model()
    tensor = _preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    screen_prob = float(probs[0])
    printout_prob = float(probs[1])
    is_screen = screen_prob >= _SCREEN_THRESHOLD
    is_printout = printout_prob >= _PRINTOUT_THRESHOLD

    if is_screen:
        label = "screen"
    elif is_printout:
        label = "printout"
    else:
        label = "live"

    return PresentationAttackResult(
        is_screen_replay=is_screen,
        is_printout=is_printout,
        screen_score=screen_prob,
        printout_score=printout_prob,
        label=label,
    )
