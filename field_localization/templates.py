"""Normalized layout templates for Stage 4 field localization."""
from __future__ import annotations

from dataclasses import dataclass

from field_localization.providers import OCRWord


@dataclass(frozen=True)
class Zone:
    """Normalized 0-1 region on the cropped ID card."""

    left: float
    top: float
    right: float
    bottom: float


FieldTemplate = dict[str, Zone]


# Conservative v1 templates. These are intentionally wider than the exact text
# areas so they tolerate mild crop/warp variation and older card layouts.
_TEMPLATES: dict[str, FieldTemplate] = {
    "legacy": {
        "id_number": Zone(0.08, 0.15, 0.45, 0.28),
        "serial_number": Zone(0.55, 0.15, 0.90, 0.28),
        "name": Zone(0.08, 0.25, 0.55, 0.42),
        "date_of_birth": Zone(0.38, 0.35, 0.68, 0.48),
        "sex": Zone(0.38, 0.42, 0.62, 0.56),
        "place_of_birth": Zone(0.35, 0.50, 0.72, 0.68),
        "date_of_issue": Zone(0.35, 0.64, 0.72, 0.80),
    },
    "maisha": {
        "id_number": Zone(0.08, 0.12, 0.45, 0.28),
        "serial_number": Zone(0.52, 0.12, 0.92, 0.28),
        "surname": Zone(0.08, 0.28, 0.58, 0.42),
        "name": Zone(0.08, 0.38, 0.62, 0.55),
        "date_of_birth": Zone(0.45, 0.35, 0.78, 0.50),
        "sex": Zone(0.45, 0.45, 0.68, 0.58),
        "date_of_issue": Zone(0.45, 0.62, 0.78, 0.78),
    },
    "passport": {
        "surname": Zone(0.05, 0.18, 0.55, 0.35),
        "name": Zone(0.05, 0.28, 0.60, 0.48),
        "nationality": Zone(0.45, 0.35, 0.75, 0.52),
        "date_of_birth": Zone(0.45, 0.45, 0.78, 0.62),
        "sex": Zone(0.70, 0.45, 0.90, 0.62),
        "id_number": Zone(0.45, 0.15, 0.90, 0.32),
        "mrz": Zone(0.02, 0.72, 0.98, 0.98),
    },
}


def get_template(id_type: str) -> FieldTemplate:
    """Return field zones for an ID type, falling back to legacy zones."""
    return _TEMPLATES.get(id_type, _TEMPLATES["legacy"])


def assign_words_to_fields(words: list[OCRWord], template: FieldTemplate) -> dict[str, list[OCRWord]]:
    """Group words by field zone using center-point inclusion."""
    grouped: dict[str, list[OCRWord]] = {field: [] for field in template}
    for word in words:
        cx = word.left + word.width / 2
        cy = word.top + word.height / 2
        for field, zone in template.items():
            if zone.left <= cx <= zone.right and zone.top <= cy <= zone.bottom:
                grouped[field].append(word)

    for field_words in grouped.values():
        field_words.sort(key=lambda w: (round(w.top, 2), w.left))
    return grouped
