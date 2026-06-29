from __future__ import annotations

import csv
import json

from field_localization.geometry import Rect, homography_full_to_canonical, transform_points, transform_rect

from scripts import convert_field_labels


def test_homography_maps_card_corners_to_canonical() -> None:
    corners = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    matrix = homography_full_to_canonical(corners)

    transformed = transform_points(corners, matrix)

    assert transformed.round(3).tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_transform_rect_from_full_image_to_canonical() -> None:
    corners = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    matrix = homography_full_to_canonical(corners)

    rect = transform_rect(Rect(0.5, 0.5, 0.7, 0.62), matrix)

    assert round(rect.left, 3) == 0.5
    assert round(rect.top, 3) == 0.5
    assert round(rect.right, 3) == 0.75
    assert round(rect.bottom, 3) == 0.7


def test_convert_field_labels_writes_canonical_csv(tmp_path) -> None:
    label_export = tmp_path / "labels.json"
    ocr_csv = tmp_path / "ocr.csv"
    out = tmp_path / "canonical.csv"
    templates_out = tmp_path / "templates.json"

    task = {
        "data": {"image": "s3://bucket/id-doc-front/2026/01/01/abc.jpg"},
        "annotations": [{
            "result": [
                {
                    "from_name": "quality",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["good_front"]},
                },
                {
                    "from_name": "id_type",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["legacy"]},
                },
                {
                    "from_name": "corners",
                    "to_name": "image",
                    "type": "polygonlabels",
                    "value": {
                        "points": [[10, 20], [90, 20], [90, 80], [10, 80]],
                        "polygonlabels": ["id_card"],
                    },
                },
                {
                    "from_name": "field",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": 50,
                        "y": 50,
                        "width": 20,
                        "height": 12,
                        "rectanglelabels": ["name"],
                    },
                },
            ]
        }],
    }
    label_export.write_text(json.dumps([task]), encoding="utf-8")

    response = {
        "textDetections": [{
            "type": "WORD",
            "detectedText": "JOHN",
            "confidence": 99.0,
            "geometry": {"boundingBox": {"left": 0.55, "top": 0.52, "width": 0.05, "height": 0.04}},
        }]
    }
    with ocr_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["s3_key", "response_summary_json"])
        writer.writeheader()
        writer.writerow({
            "s3_key": "id-doc-front/2026/01/01/abc.jpg",
            "response_summary_json": json.dumps(response),
        })

    assert convert_field_labels.main_with_args(
        ["--label-export", str(label_export), "--ocr-csv", str(ocr_csv),
         "--out", str(out), "--templates-out", str(templates_out)]
    ) == 0

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["field_name"] == "name"
    assert rows[0]["field_text"] == "JOHN"
    assert round(float(rows[0]["canonical_left"]), 3) == 0.5
    assert round(float(rows[0]["canonical_top"]), 3) == 0.5
