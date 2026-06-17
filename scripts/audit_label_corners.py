"""Audit Label Studio polygon corner order (TL -> TR -> BR -> BL).

Label Studio stores polygon vertices in click order. This script flags tasks
where that order likely does not match card-centric TL, TR, BR, BL — including
rotated IDs (order is checked in image space using geometric corner roles).

Usage:
    python scripts/audit_label_corners.py data/labels/label_studio_export.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED = ["TL", "TR", "BR", "BL"]


def canonical_corners(pts: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL rows for a 4-point quadrilateral."""
    rect = np.zeros((4, 2), dtype=np.float64)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def assign_corner_roles(pts: list[list[float]]) -> list[str]:
    """Map each clicked point to TL/TR/BR/BL by nearest geometric corner."""
    stored = np.asarray(pts, dtype=np.float64)
    canon = canonical_corners(stored)
    roles: list[str] = []
    for point in stored:
        dists = np.linalg.norm(canon - point, axis=1)
        roles.append(EXPECTED[int(np.argmin(dists))])
    return roles


def _image_name(task: dict) -> str:
    image = task.get("data", {}).get("image", "")
    return Path(image.split("?")[0]).name or str(task.get("id", "unknown"))


def _polygon_results(annotation: dict) -> list[list[list[float]]]:
    polygons: list[list[list[float]]] = []
    for item in annotation.get("result", []):
        if item.get("type") not in {"polygonlabels", "polygon"}:
            continue
        points = item.get("value", {}).get("points")
        if points and len(points) == 4:
            polygons.append(points)
    return polygons


def audit_export(export_path: Path) -> list[dict]:
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [tasks])

    findings: list[dict] = []
    for task in tasks:
        name = _image_name(task)
        task_id = task.get("id")
        annotations = task.get("annotations") or []
        if not annotations:
            continue

        for ann_idx, annotation in enumerate(annotations):
            for poly_idx, points in enumerate(_polygon_results(annotation)):
                roles = assign_corner_roles(points)
                status = "ok" if roles == EXPECTED else "wrong_order"
                if status == "wrong_order":
                    findings.append(
                        {
                            "task_id": task_id,
                            "image": name,
                            "annotation": ann_idx,
                            "polygon": poly_idx,
                            "click_order_roles": roles,
                            "expected": EXPECTED,
                            "hint": _hint(roles),
                        }
                    )
    return findings


def _hint(roles: list[str]) -> str:
    if roles == ["TL", "TR", "BL", "BR"]:
        return "Bottom corners swapped (TR/BR order may be wrong)"
    if roles == ["TR", "BR", "BL", "TL"] or roles == ["BL", "TL", "TR", "BR"]:
        return "Rotated card — corners shifted (did not start at card TL)"
    if roles == list(reversed(EXPECTED)):
        return "Reverse order (BL -> BR -> TR -> TL)"
    if roles[0] == "TL" and roles[-1] == "TR":
        return "Clockwise vs counter-clockwise mix-up"
    return "Corner order does not match TL -> TR -> BR -> BL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit polygon corner click order")
    parser.add_argument(
        "export_json",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "data" / "labels" / "label_studio_export.json",
    )
    parser.add_argument(
        "--list-ok",
        action="store_true",
        help="Also print images with correct order",
    )
    args = parser.parse_args()

    if not args.export_json.is_file():
        print(f"Export not found: {args.export_json}", file=sys.stderr)
        print(
            "\nIn Label Studio: Export -> JSON -> save to:\n"
            "  data/labels/label_studio_export.json",
            file=sys.stderr,
        )
        return 1

    tasks = json.loads(args.export_json.read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [tasks])

    total_polygons = 0
    ok_count = 0
    wrong: list[dict] = []

    for task in tasks:
        name = _image_name(task)
        for annotation in task.get("annotations") or []:
            for points in _polygon_results(annotation):
                total_polygons += 1
                roles = assign_corner_roles(points)
                if roles == EXPECTED:
                    ok_count += 1
                    if args.list_ok:
                        print(f"OK   {name}  {roles}")
                else:
                    row = {
                        "image": name,
                        "task_id": task.get("id"),
                        "click_order_roles": roles,
                        "hint": _hint(roles),
                    }
                    wrong.append(row)

    print(f"\nPolygons checked: {total_polygons}")
    print(f"Correct order:    {ok_count}")
    print(f"Need review:      {len(wrong)}\n")

    for row in wrong:
        roles = " -> ".join(row["click_order_roles"])
        print(f"REVIEW  {row['image']}")
        print(f"        task {row['task_id']}  clicked: {roles}")
        print(f"        {row['hint']}\n")

    if wrong:
        out = args.export_json.parent / "corner_order_review.json"
        out.write_text(json.dumps(wrong, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
