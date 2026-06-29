"""Field-specific parsers for Stage 4 OCR output."""
from __future__ import annotations

import re
from dataclasses import replace

from orchestration.results import ExtractedFields

from field_localization.providers import OCRWord

_DATE_RE = re.compile(r"\b(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})\b")
_ID_RE = re.compile(r"\b(\d{7,9})\b")
_SEX_RE = re.compile(r"\b(MALE|FEMALE|M|F)\b", re.IGNORECASE)
_MRZ_RE = re.compile(r"[A-Z0-9<]{30,44}")


def words_to_text(words: list[OCRWord]) -> str:
    return " ".join(word.text for word in words).strip()


def avg_confidence(words: list[OCRWord]) -> float:
    if not words:
        return 0.0
    return sum(w.confidence for w in words) / len(words)


def parse_fields(id_type: str, grouped: dict[str, list[OCRWord]]) -> ExtractedFields:
    """Parse grouped OCR words into structured fields."""
    if id_type == "passport":
        fields = _parse_passport_mrz(grouped.get("mrz", []))
        if any([fields.name, fields.surname, fields.date_of_birth, fields.sex]):
            return _fill_from_zones(fields, grouped)
    return _fill_from_zones(ExtractedFields(nationality="KENYA"), grouped)


def _fill_from_zones(fields: ExtractedFields, grouped: dict[str, list[OCRWord]]) -> ExtractedFields:
    all_text = " ".join(words_to_text(words) for words in grouped.values()).upper()

    id_text = words_to_text(grouped.get("id_number", []))
    id_number = _first_match(_ID_RE, id_text) or _first_match(_ID_RE, all_text)

    dob_text = words_to_text(grouped.get("date_of_birth", []))
    dob = _first_match(_DATE_RE, dob_text) or _first_match(_DATE_RE, all_text)

    issue_text = words_to_text(grouped.get("date_of_issue", []))
    issue = _first_match(_DATE_RE, issue_text)

    sex_text = words_to_text(grouped.get("sex", []))
    sex = _normalise_sex(_first_match(_SEX_RE, sex_text) or _first_match(_SEX_RE, all_text))

    return replace(
        fields,
        id_number=fields.id_number or id_number,
        name=fields.name or _clean_name(words_to_text(grouped.get("name", []))),
        surname=fields.surname or _clean_name(words_to_text(grouped.get("surname", []))),
        sex=fields.sex or sex,
        date_of_birth=fields.date_of_birth or dob,
        date_of_issue=fields.date_of_issue or issue,
        place_of_birth=fields.place_of_birth or _clean_name(
            words_to_text(grouped.get("place_of_birth", []))
        ),
        serial_number=fields.serial_number or _first_match(
            _ID_RE, words_to_text(grouped.get("serial_number", []))
        ),
    )


def _parse_passport_mrz(words: list[OCRWord]) -> ExtractedFields:
    text = words_to_text(words).replace(" ", "").upper()
    lines = _MRZ_RE.findall(text)
    if len(lines) < 2:
        return ExtractedFields()
    line1, line2 = lines[0], lines[1]
    name_raw = line1[5:] if len(line1) > 5 else ""
    parts = name_raw.replace("<<", "|").replace("<", " ").split("|")
    dob = None
    sex = None
    detail = re.search(r"[A-Z<]{3}(\d{6})\d?([MF<])", line2)
    if detail:
        raw = detail.group(1)
        dob = f"{raw[4:6]}/{raw[2:4]}/{raw[:2]}"
        sex = _normalise_sex(detail.group(2))
    elif len(line2) >= 20:
        raw = line2[13:19]
        if raw.isdigit():
            dob = f"{raw[4:6]}/{raw[2:4]}/{raw[:2]}"
        sex = _normalise_sex(line2[20] if len(line2) > 20 else None)
    return ExtractedFields(
        surname=_clean_name(parts[0]) if parts else None,
        name=_clean_name(parts[1]) if len(parts) > 1 else None,
        sex=sex,
        nationality=line1[2:5].replace("<", "") if len(line1) >= 5 else None,
        date_of_birth=dob,
    )


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _normalise_sex(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    if upper == "M":
        return "MALE"
    if upper == "F":
        return "FEMALE"
    if upper in {"MALE", "FEMALE"}:
        return upper
    return None


def _clean_name(value: str) -> str | None:
    cleaned = re.sub(r"[^A-Z '\-]", " ", value.upper())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None
