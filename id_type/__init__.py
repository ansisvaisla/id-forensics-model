"""Stage 4 — ID Type Classification.

Entry point: run(image) -> IdTypeResult

v1: stub returning 'unknown' until model is trained.
v2: EfficientNet-B0 or ResNet-18 fine-tuned on legacy/maisha/other labels.
    Requires deskewed (Stage 1 output) image for reliable classification.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestration.results import IdTypeResult

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "stage4_id_type" / "best.pt"
)

KNOWN_TYPES = (
    "legacy",
    "maisha",
    "huduma",
    "passport",
    "driving_licence",
    "foreign_document",
    "unknown_id",
)

_model = None


def _load_model():
    import torch  # type: ignore
    import timm  # type: ignore

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(KNOWN_TYPES))
    state = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


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


def run(image: np.ndarray) -> IdTypeResult:
    """Classify ID type from deskewed card image.

    Args:
        image: BGR numpy array — should be deskewed output from Stage 1.

    Returns:
        IdTypeResult with id_type and confidence.
        Returns 'unknown' if model not yet trained.
    """
    if not MODEL_PATH.is_file():
        return IdTypeResult(id_type="unknown", confidence=0.0)

    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    model = _get_model()
    tensor = _preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return IdTypeResult(
        id_type=KNOWN_TYPES[idx],
        confidence=float(probs[idx]),
    )
