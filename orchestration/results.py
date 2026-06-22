"""Shared dataclasses and metadata flags propagated across all pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IdCropResult:
    """Output of Stage 1 — ID Crop."""
    cropped_image: Optional[object]  # numpy ndarray or None
    is_partial_document: bool
    corners_detected: int  # 0–4
    label: str  # e.g. 'selfie_instead_of_document', 'id_card', 'no_id_detected'
    confidence: float
    # Bounding box in the *original* image as (x%, y%, w%, h%) — 0-100 percentages.
    # Used by batch_label.py to emit rectanglelabels in Label Studio JSON.
    bbox_orig: Optional[tuple[float, float, float, float]] = None


@dataclass
class PresentationAttackResult:
    """Output of Stage 2 — Presentation Attack Detection."""
    is_screen_replay: bool
    is_printout: bool
    screen_score: float   # probability 0–1; higher = more likely screen
    printout_score: float  # probability 0–1; placeholder until v2 model exists
    label: str  # 'screen' | 'printout' | 'live'


@dataclass
class TamperingResult:
    """Output of Stage 3 — Injection & Digital Tampering Detection."""
    is_tampered: bool
    ela_score: float      # normalised ELA residual energy 0–1
    exif_suspicious: bool  # True if EXIF lacks expected camera sensor fields
    label: str  # 'tampered' | 'clean' | 'unknown'


@dataclass
class IdTypeResult:
    """Output of Stage 4 — ID Type Classification."""
    id_type: str  # legacy | maisha | huduma | passport | driving_licence | foreign_document | unknown_id | unknown
    confidence: float


@dataclass
class ExtractedFields:
    """Structured fields parsed from OCR output (Stage 5)."""
    name: Optional[str] = None
    surname: Optional[str] = None
    sex: Optional[str] = None
    nationality: Optional[str] = None
    id_number: Optional[str] = None
    date_of_birth: Optional[str] = None


@dataclass
class FieldExtractResult:
    """Output of Stage 5 — Field Text Recognizer."""
    extracted_fields: ExtractedFields
    field_extraction_confidence: float  # average Textract confidence 0–1
    label: str  # 'extracted' | 'partial' | 'failed'


@dataclass
class PipelineResult:
    """Aggregated output of all pipeline stages (Stage 7 — Orchestration)."""
    # Stage flags
    is_partial_document: bool = False
    is_screen_replay: bool = False
    is_printout: bool = False
    is_tampered: bool = False

    # Detailed results per stage (None if stage was skipped or failed)
    crop: Optional[IdCropResult] = None
    presentation_attack: Optional[PresentationAttackResult] = None
    tampering: Optional[TamperingResult] = None
    id_type: Optional[IdTypeResult] = None
    field_extract: Optional[FieldExtractResult] = None

    # Convenience accessors
    id_type_label: str = "unknown"
    extracted_fields: ExtractedFields = field(default_factory=ExtractedFields)
    field_extraction_confidence: float = 0.0
    label: str = "unknown"

    # Risk tier from decision matrix
    risk_tier: str = "unknown"  # 'high_fraud' | 'garbage_photo' | 'user_error' | 'clean'
