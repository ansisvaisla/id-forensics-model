"""Stage 2 — Presentation Attack Detection.

Entry point: run(image) -> PresentationAttackResult

Screen replay: EfficientNet-B0 fine-tuned on screen vs live labels.
Printout: deferred to v2 — returns is_printout=False, printout_score=0.0.

Shadow mode: exceptions are caught by orchestration layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestration.results import PresentationAttackResult

SCREEN_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "stage2_screen" / "best.pt"
)
_SCREEN_THRESHOLD = 0.55  # tuned from threshold sweep: best F1 0.897 at 0.55 vs 0.875 at 0.50

_screen_model = None


def _load_screen_model():
    """Lazy-load screen classifier. Raises if model file missing."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torchvision.models import efficientnet_b0  # type: ignore

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
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
        PresentationAttackResult with is_screen_replay and screen_score.
        is_printout is always False in v1 (deferred).
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
    # class 0 = screen, class 1 = live (mirrors split_yolo_dataset labels)
    screen_prob = float(probs[0])
    is_screen = screen_prob >= _SCREEN_THRESHOLD

    return PresentationAttackResult(
        is_screen_replay=is_screen,
        is_printout=False,
        screen_score=screen_prob,
        printout_score=0.0,
        label="screen" if is_screen else "live",
    )
