"""Stage 5 — Field Text Recognizer.

Entry point: run(image_bytes, id_type) -> FieldExtractResult

Uses AWS Textract to get word bounding boxes from the deskewed ID image,
then applies a position-based parser per id_type to extract:
  name, surname, sex, nationality, id_number, date_of_birth

v1 strategy — no field-location labeling needed:
  - legacy:   position heuristics (name top-center, ID number right-middle, DOB lower area)
  - maisha:   surname line 1, given name line 2, ID number chip zone
  - passport: MRZ bottom two lines → standard MRZ regex parser
  - fallback: keyword proximity search for known field labels

Shadow mode: exceptions caught by orchestration layer.
"""
from __future__ import annotations

import re
from typing import Optional

from orchestration.results import ExtractedFields, FieldExtractResult

# MRZ line regex for ICAO 9303 TD3 (passport)
_MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{44}$")
_MRZ_DOB_RE = re.compile(r"^\d{6}[MF<]\d{6}")  # within MRZ line 2

# Common field keyword patterns for fallback parser
_KEYWORD_PATTERNS = {
    "id_number": re.compile(r"\b(\d{7,9})\b"),
    "sex": re.compile(r"\b(MALE|FEMALE|M|F)\b", re.IGNORECASE),
    "date_of_birth": re.compile(r"\b(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})\b"),
}


def _call_textract(image_bytes: bytes) -> list[dict]:
    """Call AWS Textract DetectDocumentText; return list of word blocks with geometry."""
    import boto3  # type: ignore
    client = boto3.client("textract")
    response = client.detect_document_text(Document={"Bytes": image_bytes})
    blocks = []
    for block in response.get("Blocks", []):
        if block.get("BlockType") == "WORD":
            geo = block.get("Geometry", {}).get("BoundingBox", {})
            blocks.append({
                "text": block.get("Text", ""),
                "confidence": block.get("Confidence", 0.0) / 100.0,
                "left": geo.get("Left", 0.0),
                "top": geo.get("Top", 0.0),
                "width": geo.get("Width", 0.0),
                "height": geo.get("Height", 0.0),
            })
    return blocks


def _words_in_zone(
    blocks: list[dict],
    top_min: float,
    top_max: float,
    left_min: float = 0.0,
    left_max: float = 1.0,
) -> list[str]:
    """Collect word texts within a normalised bounding zone."""
    return [
        b["text"]
        for b in blocks
        if top_min <= b["top"] <= top_max and left_min <= b["left"] <= left_max
    ]


def _avg_confidence(blocks: list[dict]) -> float:
    if not blocks:
        return 0.0
    return sum(b["confidence"] for b in blocks) / len(blocks)


def _parse_legacy(blocks: list[dict]) -> ExtractedFields:
    """Position parser for Kenyan Legacy ID."""
    full_text = " ".join(b["text"] for b in blocks).upper()
    # Name: typically top 20% of card, centre zone
    name_words = _words_in_zone(blocks, 0.0, 0.20, 0.15, 0.85)
    # ID number: right side, middle band
    id_words = _words_in_zone(blocks, 0.35, 0.65, 0.55, 1.0)
    id_number = next(
        (w for w in id_words if re.match(r"^\d{7,9}$", w)), None
    )
    # DOB: lower band
    dob_words = _words_in_zone(blocks, 0.60, 0.85)
    dob = next(
        (w for w in dob_words if _KEYWORD_PATTERNS["date_of_birth"].match(w)), None
    )
    # Sex: near DOB
    sex_match = _KEYWORD_PATTERNS["sex"].search(full_text)
    sex = sex_match.group(1).upper() if sex_match else None

    return ExtractedFields(
        name=" ".join(name_words[:2]) if name_words else None,
        surname=name_words[0] if name_words else None,
        sex=sex,
        nationality="KENYA",  # always Kenya for legacy
        id_number=id_number,
        date_of_birth=dob,
    )


def _parse_maisha(blocks: list[dict]) -> ExtractedFields:
    """Position parser for Kenyan Maisha ID."""
    # Surname line 1 (top ~25%)
    surname_words = _words_in_zone(blocks, 0.0, 0.25, 0.10, 0.90)
    # Given name line 2 (25–45%)
    name_words = _words_in_zone(blocks, 0.25, 0.45, 0.10, 0.90)
    # ID number chip zone (typically bottom-right)
    id_words = _words_in_zone(blocks, 0.55, 0.85, 0.50, 1.0)
    id_number = next(
        (w for w in id_words if re.match(r"^\d{7,9}$", w)), None
    )
    full_text = " ".join(b["text"] for b in blocks).upper()
    dob_match = _KEYWORD_PATTERNS["date_of_birth"].search(full_text)
    sex_match = _KEYWORD_PATTERNS["sex"].search(full_text)

    return ExtractedFields(
        surname=" ".join(surname_words) if surname_words else None,
        name=" ".join(name_words) if name_words else None,
        sex=sex_match.group(1).upper() if sex_match else None,
        nationality="KENYA",
        id_number=id_number,
        date_of_birth=dob_match.group(0) if dob_match else None,
    )


def _parse_passport(blocks: list[dict]) -> ExtractedFields:
    """MRZ parser for ICAO TD3 passports (bottom two lines, 44 chars each)."""
    # Collect bottom 20% words in left-to-right order
    mrz_words = sorted(
        [b for b in blocks if b["top"] >= 0.75],
        key=lambda b: (round(b["top"], 1), b["left"]),
    )
    mrz_text = "".join(b["text"] for b in mrz_words).replace(" ", "").upper()
    # Try to extract two 44-char MRZ lines
    if len(mrz_text) >= 88:
        line1, line2 = mrz_text[:44], mrz_text[44:88]
        # Line 2: DOB at positions 13-18, sex at 20, nationality from line 1 pos 2-5
        dob_raw = line2[13:19] if len(line2) >= 19 else None
        dob = f"{dob_raw[4:6]}/{dob_raw[2:4]}/{dob_raw[:2]}" if dob_raw else None
        sex_char = line2[20] if len(line2) > 20 else None
        sex = {"M": "MALE", "F": "FEMALE"}.get(sex_char or "", None)
        nat = line1[2:5].replace("<", "") if len(line1) >= 5 else None
        # Name from line 1 after nationality (pos 5+)
        name_raw = line1[5:].replace("<<", " | ").replace("<", "").strip()
        parts = name_raw.split("|") if "|" in name_raw else [name_raw, ""]
        return ExtractedFields(
            surname=parts[0].strip() if parts else None,
            name=parts[1].strip() if len(parts) > 1 else None,
            sex=sex,
            nationality=nat,
            id_number=None,
            date_of_birth=dob,
        )
    # Fallback: keyword-based
    return _parse_fallback(blocks)


def _parse_fallback(blocks: list[dict]) -> ExtractedFields:
    """Keyword proximity fallback for unknown id_type."""
    full_text = " ".join(b["text"] for b in blocks).upper()
    id_match = _KEYWORD_PATTERNS["id_number"].search(full_text)
    dob_match = _KEYWORD_PATTERNS["date_of_birth"].search(full_text)
    sex_match = _KEYWORD_PATTERNS["sex"].search(full_text)
    return ExtractedFields(
        id_number=id_match.group(1) if id_match else None,
        date_of_birth=dob_match.group(0) if dob_match else None,
        sex=sex_match.group(1).upper() if sex_match else None,
    )


_PARSERS = {
    "legacy": _parse_legacy,
    "maisha": _parse_maisha,
    "passport": _parse_passport,
}


def run(image_bytes: bytes, id_type: str = "unknown") -> FieldExtractResult:
    """Extract structured fields from an ID image via AWS Textract.

    Args:
        image_bytes: raw JPEG bytes of the deskewed ID card.
        id_type: output of Stage 4 — determines which position parser to use.

    Returns:
        FieldExtractResult with extracted_fields and confidence score.
    """
    blocks = _call_textract(image_bytes)
    if not blocks:
        return FieldExtractResult(
            extracted_fields=ExtractedFields(),
            field_extraction_confidence=0.0,
            label="failed",
        )

    parser = _PARSERS.get(id_type, _parse_fallback)
    fields = parser(blocks)
    avg_conf = _avg_confidence(blocks)

    filled = sum(
        1 for v in [fields.name, fields.surname, fields.id_number, fields.date_of_birth]
        if v is not None
    )
    label = "extracted" if filled >= 3 else ("partial" if filled >= 1 else "failed")

    return FieldExtractResult(
        extracted_fields=fields,
        field_extraction_confidence=avg_conf,
        label=label,
    )
