"""Deprecated compatibility module for Stage 1 — Quality Gate.

Entry point: run(image) -> QualityGateResult

8-class EfficientNet-B0:
    screen (0) | printout (1) | selfie (2) | back (3) | garbage (4)
    good_front (5) | partial (6) | blurry (7)

Decision strategy — binary rejection:
    Instead of argmax, we sum the probabilities of all 5 reject classes.
    If that combined reject_score exceeds REJECT_THRESHOLD the image is flagged,
    regardless of which single class has the highest softmax output.
    This prevents edge cases like a screen replay being labelled "blurry" (argmax)
    while still carrying a significant combined screen+printout probability.

Shadow mode: exceptions are caught by the orchestration layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestration.results import QualityGateResult

SCREEN_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "stage1_quality_gate" / "best.pt"
)
LEGACY_SCREEN_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "stage2_screen" / "best.pt"
)

_NUM_CLASSES = 8
_CLASS_NAMES = ("screen", "printout", "selfie", "back", "garbage", "good_front", "partial", "blurry")
_REJECT_INDICES = (0, 1, 2, 3, 4)   # screen, printout, selfie, back, garbage
_ACCEPT_INDICES = (5, 6, 7)          # good_front, partial, blurry

# Reject if the combined probability of all 5 reject classes exceeds this.
# Lowered to 0.25 because the retrained model pushes screen probability lower
# on hard cases (screen-as-partial/blurry) — a stricter threshold recaptures them.
# Tune upward (more permissive) or downward (stricter) as needed.
_REJECT_THRESHOLD = 0.25

_screen_model = None


def _load_screen_model():
    """Lazy-load Stage 1 classifier. Raises if model file missing."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torchvision.models import efficientnet_b0  # type: ignore

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, _NUM_CLASSES)
    state = torch.load(str(_model_path()), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_screen_model():
    global _screen_model
    if _screen_model is None:
        _screen_model = _load_screen_model()
    return _screen_model


def _model_path() -> Path:
    """Prefer new Stage 1 path, but keep old Stage 2 artifacts usable."""
    return SCREEN_MODEL_PATH if SCREEN_MODEL_PATH.is_file() else LEGACY_SCREEN_MODEL_PATH


def _preprocess(image: np.ndarray) -> "torch.Tensor":
    """Resize and normalise image for EfficientNet-B0."""
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
    """Classify image through the quality gate.

    Args:
        image: BGR numpy array (raw upload — runs before id_crop).

    Returns:
        QualityGateResult with is_live flag and per-class confidence.
        is_live=True  → image proceeds to id_crop + downstream stages.
        is_live=False → image is rejected (screen / printout / selfie / back / garbage).
    """
    if not _model_path().is_file():
        return QualityGateResult(
            label="model_not_trained",
            confidence=0.0,
            is_live=True,   # fail-open: don't block if model missing
            is_screen_replay=False,
            is_printout=False,
        )

    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    try:
        model = _get_screen_model()
    except RuntimeError:
        return QualityGateResult(
            label="model_not_trained",
            confidence=0.0,
            is_live=True,   # fail-open: don't block if legacy weights are incompatible
            is_screen_replay=False,
            is_printout=False,
        )
    tensor = _preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    probs_list = probs.tolist()
    argmax_idx = int(probs.argmax())
    argmax_label = _CLASS_NAMES[argmax_idx]
    argmax_conf = float(probs[argmax_idx])

    # Binary rejection: sum all reject-class probabilities.
    reject_score = sum(probs_list[i] for i in _REJECT_INDICES)
    is_rejected = reject_score >= _REJECT_THRESHOLD

    # For is_screen_replay / is_printout flags preserve individual scores.
    screen_prob = probs_list[0]
    printout_prob = probs_list[1]

    if is_rejected:
        # Use the argmax label as the rejection reason (for logging/debugging).
        label = argmax_label if argmax_idx in _REJECT_INDICES else "reject_combined"
    else:
        label = argmax_label  # good_front / partial / blurry

    return QualityGateResult(
        label=label,
        confidence=argmax_conf if not is_rejected else reject_score,
        is_live=not is_rejected,
        is_screen_replay=argmax_label == "screen" or screen_prob >= 0.4,
        is_printout=argmax_label == "printout" or printout_prob >= 0.4,
    )
