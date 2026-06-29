from __future__ import annotations

import numpy as np

import field_localization
from field_localization.providers import OCRWord
from field_localization.templates import assign_words_to_fields, get_template


def _word(text: str, left: float, top: float, width: float = 0.05) -> OCRWord:
    return OCRWord(text=text, confidence=0.9, left=left, top=top, width=width, height=0.03)


def test_assign_words_to_legacy_template() -> None:
    words = [
        _word("230820891", 0.22, 0.20),
        _word("MICHAEL", 0.12, 0.32),
        _word("MALE", 0.45, 0.49),
    ]

    grouped = assign_words_to_fields(words, get_template("legacy"))

    assert [w.text for w in grouped["id_number"]] == ["230820891"]
    assert [w.text for w in grouped["name"]] == ["MICHAEL"]
    assert [w.text for w in grouped["sex"]] == ["MALE"]


def test_run_with_injected_aws_words_extracts_legacy_fields() -> None:
    words = [
        _word("230820891", 0.22, 0.20),
        _word("MICHAEL", 0.12, 0.32),
        _word("WAITHAKA", 0.25, 0.32),
        _word("1975", 0.46, 0.40),
        _word("MALE", 0.45, 0.49),
        _word("24.10.2012", 0.46, 0.70),
    ]

    result = field_localization.run(np.zeros((1, 1, 3), dtype=np.uint8), "legacy", words)

    assert result.label == "extracted"
    assert result.extracted_fields.id_number == "230820891"
    assert result.extracted_fields.name == "MICHAEL WAITHAKA"
    assert result.extracted_fields.sex == "MALE"
    assert result.extracted_fields.date_of_issue == "24.10.2012"


def test_passport_mrz_parses_name_and_dob() -> None:
    words = [
        _word("P<KENMUTUA<<ANN<<<<<<<<<<<<<<<<<<<<<<<<", 0.05, 0.80, 0.8),
        _word("A1234567<8KEN9001012F3001012<<<<<<<<<<<<<<02", 0.05, 0.86, 0.8),
    ]

    result = field_localization.run(np.zeros((1, 1, 3), dtype=np.uint8), "passport", words)

    assert result.extracted_fields.surname == "MUTUA"
    assert result.extracted_fields.name == "ANN"
    assert result.extracted_fields.sex == "FEMALE"
    assert result.extracted_fields.date_of_birth == "01/01/90"
