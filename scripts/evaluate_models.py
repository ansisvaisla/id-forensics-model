"""Unified model evaluation script for all pipeline stages.

Produces per-run output under data/eval/<stage>/<YYYYMMDD_HHMMSS>/:
    metrics.json        — all numeric metrics
    predictions.csv     — per-image: truth, prediction, confidence
    report.txt          — human-readable summary
    viz/                — annotated image thumbnails
        correct/        — green-bordered correct predictions
        wrong/          — red-bordered misclassifications
    confusion_matrix.png   (stage1, stage3 only)
    confidence_hist.png    (stage1, stage3 only)
    per_class_grid.png     (stage3 only — sample crops per class)

Usage:
    python scripts/evaluate_models.py                      # all stages on val split
    python scripts/evaluate_models.py --stage stage1 --split test
    python scripts/evaluate_models.py --stage stage2 --max-viz 30
    python scripts/evaluate_models.py --stage stage3

Stages:
    stage1  — EfficientNet-B0 quality gate classifier
    stage2  — YOLOv8 Pose corner detection (key-point error in pixels)
    stage3  — EfficientNet-B0 ID type classifier (7 classes)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Default eval root — override with set_eval_root() or --output-dir
_EVAL_ROOT = PROJECT_ROOT / "data" / "eval"


def set_eval_root(path: Path | str) -> None:
    """Override where evaluation outputs are written (e.g. Google Drive on Colab)."""
    global _EVAL_ROOT
    _EVAL_ROOT = Path(path)


def get_eval_root() -> Path:
    return _EVAL_ROOT

# Model paths
STAGE1_MODEL = PROJECT_ROOT / "models" / "stage1_quality_gate" / "best.pt"
STAGE1_LEGACY_MODEL = PROJECT_ROOT / "models" / "stage2_screen" / "best.pt"
STAGE2_MODEL = PROJECT_ROOT / "models" / "stage2_corners" / "weights" / "best.pt"
STAGE2_LEGACY_MODEL = PROJECT_ROOT / "models" / "stage1_corners" / "weights" / "best.pt"
STAGE3_MODEL = PROJECT_ROOT / "models" / "stage3_id_type" / "best.pt"
STAGE3_LEGACY_MODEL = PROJECT_ROOT / "models" / "stage4_id_type" / "best.pt"

# Data roots
STAGE1_DATA = PROJECT_ROOT / "data" / "yolo" / "screen"
STAGE2_DATA = PROJECT_ROOT / "data" / "yolo" / "corners"
STAGE3_DATA = PROJECT_ROOT / "data" / "id_type"

STAGE3_CLASSES = (
    "legacy", "maisha", "huduma", "passport",
    "driving_licence", "foreign_document", "unknown_id",
)
STAGE1_CLASSES = (
    "screen", "printout", "selfie", "back",
    "garbage", "good_front", "partial", "blurry",
)
_NUM_STAGE1 = len(STAGE1_CLASSES)

THUMB = 280
BORDER = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _first_existing(*paths: Path) -> Path:
    """Return first existing path, or the preferred path if none exist."""
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_run_dir(stage: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = _EVAL_ROOT / stage / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / "viz" / "correct").mkdir(parents=True, exist_ok=True)
    (d / "viz" / "wrong").mkdir(parents=True, exist_ok=True)
    return d


def _thumbnail(img: np.ndarray, size: int = THUMB) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))


def _bordered(img: np.ndarray, correct: bool, label_text: str) -> np.ndarray:
    """Wrap image in green (correct) or red (wrong) border with label strip."""
    thumb = _thumbnail(img)
    th, tw = thumb.shape[:2]
    color = (0, 180, 0) if correct else (0, 0, 220)
    canvas = np.full((th + 2 * BORDER + 38, tw + 2 * BORDER, 3), 25, np.uint8)
    canvas[BORDER:BORDER + th, BORDER:BORDER + tw] = thumb
    cv2.rectangle(canvas, (0, 0), (tw + 2 * BORDER - 1, th + 2 * BORDER + 37), color, BORDER)
    cv2.putText(canvas, label_text, (BORDER, th + 2 * BORDER + 28),
                FONT, 0.38, (220, 220, 220), 1)
    return canvas


def _save_confusion_matrix(
    y_true: list[str], y_pred: list[str],
    classes: tuple[str, ...], out_path: Path
) -> None:
    """Save confusion matrix as a PNG image."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1

    cell = 56
    font_scale = 0.4
    img_h = cell * (n + 1)
    img_w = cell * (n + 1) + 60
    canvas = np.full((img_h, img_w, 3), 245, np.uint8)

    max_val = cm.max() if cm.max() > 0 else 1
    short = [c[:8] for c in classes]

    # Column headers (predicted)
    for j, name in enumerate(short):
        cv2.putText(canvas, name, (60 + j * cell + 2, 20), FONT, font_scale, (60, 60, 60), 1)

    # Rows
    for i, name in enumerate(short):
        y = 38 + i * cell
        # Row label (true)
        cv2.putText(canvas, name, (2, y + cell // 2), FONT, font_scale, (60, 60, 60), 1)
        for j in range(n):
            x = 60 + j * cell
            val = cm[i][j]
            intensity = int(255 * (1.0 - val / max_val))
            bg = (intensity, intensity, 255) if i == j else (255, intensity, intensity) if val > 0 else (245, 245, 245)
            cv2.rectangle(canvas, (x, y), (x + cell - 2, y + cell - 2), bg, -1)
            cv2.rectangle(canvas, (x, y), (x + cell - 2, y + cell - 2), (180, 180, 180), 1)
            cv2.putText(canvas, str(val), (x + cell // 2 - 8, y + cell // 2 + 5),
                        FONT, 0.45, (20, 20, 20), 1)

    cv2.imwrite(str(out_path), canvas)


def _save_confidence_hist(confidences: list[float], correct_flags: list[bool], out_path: Path) -> None:
    """Save a simple confidence histogram as a PNG (correct vs wrong split)."""
    bins = 10
    bin_correct = [0] * bins
    bin_wrong = [0] * bins
    for conf, ok in zip(confidences, correct_flags):
        b = min(int(conf * bins), bins - 1)
        if ok:
            bin_correct[b] += 1
        else:
            bin_wrong[b] += 1

    W, H, pad = 500, 220, 40
    canvas = np.full((H + pad, W + pad, 3), 245, np.uint8)
    max_count = max(max(bin_correct + bin_wrong), 1)
    bar_w = (W - 10) // bins

    for i in range(bins):
        x = pad + i * bar_w
        # Wrong (red) on top of correct (green)
        hc = int((bin_correct[i] / max_count) * (H - 20))
        hw = int((bin_wrong[i] / max_count) * (H - 20))
        if hc > 0:
            cv2.rectangle(canvas, (x, pad + H - hc), (x + bar_w - 3, pad + H), (0, 170, 0), -1)
        if hw > 0:
            cv2.rectangle(canvas, (x, pad + H - hc - hw), (x + bar_w - 3, pad + H - hc), (0, 0, 200), -1)
        label = f"{i/bins:.1f}"
        cv2.putText(canvas, label, (x, pad + H + 14), FONT, 0.3, (80, 80, 80), 1)

    cv2.putText(canvas, "Confidence distribution (green=correct, red=wrong)", (pad, 20),
                FONT, 0.4, (40, 40, 40), 1)
    cv2.imwrite(str(out_path), canvas)


def _save_per_class_grid(
    class_samples: dict[str, list[np.ndarray]],
    out_path: Path,
    n_per_class: int = 6,
) -> None:
    """Save a grid with up to n_per_class sample crops per ID type class."""
    classes = [c for c in STAGE3_CLASSES if c in class_samples and class_samples[c]]
    if not classes:
        return

    cell_h, cell_w = 120, 190
    label_h = 20
    n_cols = n_per_class
    n_rows = len(classes)
    canvas = np.full(
        (n_rows * (cell_h + label_h) + 10, n_cols * cell_w + 100, 3), 30, np.uint8
    )

    for row_i, cls in enumerate(classes):
        y = row_i * (cell_h + label_h) + 5
        cv2.putText(canvas, cls[:12], (4, y + cell_h // 2), FONT, 0.4, (200, 200, 200), 1)
        samples = class_samples[cls][:n_per_class]
        for col_i, img in enumerate(samples):
            x = 100 + col_i * cell_w
            thumb = cv2.resize(img, (cell_w - 4, cell_h - 4))
            canvas[y:y + cell_h - 4, x:x + cell_w - 4] = thumb

    cv2.imwrite(str(out_path), canvas)


# ── Stage 2 — Corners ─────────────────────────────────────────────────────────

def evaluate_stage2(split: str = "val", max_viz: int = 20, tolerance_pct: float = 0.05) -> dict:
    model_path = _first_existing(STAGE2_MODEL, STAGE2_LEGACY_MODEL)
    if not model_path.is_file():
        print(f"Stage 2 model not found: {model_path}", file=sys.stderr)
        return {}

    from ultralytics import YOLO  # type: ignore
    model = YOLO(str(model_path))

    img_dir = STAGE2_DATA / "images" / split
    lbl_dir = STAGE2_DATA / "labels" / split
    run_dir = _make_run_dir("stage2")

    distances: list[float] = []
    no_detection = 0
    within_tol = 0
    viz_count = 0
    rows: list[dict] = []

    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        parts = lbl_path.read_text().strip().split()
        vals = [float(v) for v in parts[1:]]

        if len(vals) == 16:
            true_pts = np.array([[vals[4 + i*3] * w, vals[5 + i*3] * h] for i in range(4)], np.float32)
        elif len(vals) == 8:
            true_pts = np.array([[vals[i*2] * w, vals[i*2+1] * h] for i in range(4)], np.float32)
        else:
            continue

        results = model(img, verbose=False)
        pred_pts = None
        if results:
            r = results[0]
            if hasattr(r, "keypoints") and r.keypoints is not None and len(r.keypoints) > 0:
                kpts = r.keypoints[0].xy
                if kpts is not None and len(kpts) > 0:
                    pred_pts = kpts[0].cpu().numpy()
            elif hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                pred_pts = r.obb[0].xyxyxyxy[0].cpu().numpy().reshape(4, 2)

        if pred_pts is None or len(pred_pts) < 4:
            no_detection += 1
            rows.append({"stem": img_path.stem, "detected": False, "dist_px": None})
            continue

        best_dist = float("inf")
        best_perm = list(range(4))
        for perm in permutations(range(4)):
            d = sum(math.hypot(pred_pts[perm[i], 0] - true_pts[i, 0],
                               pred_pts[perm[i], 1] - true_pts[i, 1]) for i in range(4)) / 4
            if d < best_dist:
                best_dist, best_perm = d, list(perm)

        diag = math.hypot(w, h)
        ok = best_dist < tolerance_pct * diag
        if ok:
            within_tol += 1
        distances.append(best_dist)
        rows.append({"stem": img_path.stem, "detected": True, "dist_px": round(best_dist, 2), "within_tol": ok})

        if viz_count < max_viz:
            vis = img.copy()
            matched = pred_pts[best_perm]
            for i in range(4):
                tx, ty = int(true_pts[i, 0]), int(true_pts[i, 1])
                px, py = int(matched[i, 0]), int(matched[i, 1])
                cv2.line(vis, (tx, ty), (px, py), (0, 255, 255), 2)
                cv2.circle(vis, (tx, ty), 7, (0, 255, 0), -1)
                cv2.circle(vis, (px, py), 7, (0, 0, 255), -1)
                cv2.putText(vis, f"{math.hypot(px-tx,py-ty):.0f}", (tx + 5, ty - 5), FONT, 0.42, (255,255,255), 1)
            tag = "OK" if ok else "BAD"
            cv2.putText(vis, f"mean={best_dist:.1f}px {tag}", (10, 30), FONT, 0.8, (0,255,255), 2)

            # Draw predicted bbox rectangle on original
            bx1 = max(0, int(pred_pts[:, 0].min()))
            by1 = max(0, int(pred_pts[:, 1].min()))
            bx2 = min(img.shape[1], int(pred_pts[:, 0].max()))
            by2 = min(img.shape[0], int(pred_pts[:, 1].max()))
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 140, 0), 2)

            # Pipeline crop (what actually goes to Stage 3)
            sys.path.insert(0, str(PROJECT_ROOT))
            try:
                import id_crop as _id_crop_mod
                crop_result = _id_crop_mod.run(img)
                pipeline_crop = crop_result.cropped_image
                crop_label = crop_result.label
            except Exception:
                pipeline_crop = img[by1:by2, bx1:bx2]
                crop_label = "raw_bbox"

            # Side-by-side: original+overlay | pipeline crop
            thumb_orig = _thumbnail(vis, 400)
            if pipeline_crop is not None and pipeline_crop.size > 0:
                thumb_crop = _thumbnail(pipeline_crop, 400)
            else:
                thumb_crop = np.zeros((100, 200, 3), np.uint8)

            h1, w1 = thumb_orig.shape[:2]
            h2, w2 = thumb_crop.shape[:2]
            max_h = max(h1, h2)
            pad1 = np.full((max_h - h1, w1, 3), 60, np.uint8)
            pad2 = np.full((max_h - h2, w2, 3), 60, np.uint8)
            # Draw dotted line at crop boundary so padding is obvious
            if max_h - h2 > 0:
                cv2.line(pad2, (0, 0), (w2, 0), (0, 180, 255), 2)
            col1 = np.vstack([thumb_orig, pad1])
            col2 = np.vstack([thumb_crop, pad2])
            divider = np.full((max_h, 4, 3), 80, np.uint8)
            side_by_side = np.hstack([col1, divider, col2])

            # Label strip at bottom
            strip = np.full((28, side_by_side.shape[1], 3), 15, np.uint8)
            cv2.putText(strip, f"CORNERS mean={best_dist:.1f}px {tag}", (4, 20), FONT, 0.45, (0,255,255), 1)
            cv2.putText(strip, f"CROP: {crop_label}", (w1 + 10, 20), FONT, 0.45, (255,200,0), 1)
            final = np.vstack([side_by_side, strip])

            subfolder = "correct" if ok else "wrong"
            cv2.imwrite(str(run_dir / "viz" / subfolder / f"{img_path.stem}.jpg"), final)
            viz_count += 1

    n = len(distances)
    total = n + no_detection
    mean_dist = float(np.mean(distances)) if distances else 0.0
    median_dist = float(np.median(distances)) if distances else 0.0
    pct_within = within_tol / total * 100 if total else 0.0

    metrics = {
        "stage": "stage2",
        "split": split,
        "model": str(model_path),
        "total": total,
        "detected": n,
        "no_detection": no_detection,
        "detection_rate_pct": round(n / total * 100, 2) if total else 0,
        "mean_corner_dist_px": round(mean_dist, 2),
        "median_corner_dist_px": round(median_dist, 2),
        "p90_dist_px": round(float(np.percentile(distances, 90)), 2) if distances else 0,
        "within_tolerance_pct": round(pct_within, 2),
        "tolerance": f"{int(tolerance_pct*100)}% of diagonal",
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w2 = csv.DictWriter(f, fieldnames=["stem", "detected", "dist_px", "within_tol"])
        w2.writeheader()
        w2.writerows(rows)

    report = _stage2_report(metrics)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nOutputs: {run_dir}")
    return metrics


def _stage2_report(m: dict) -> str:
    lines = [
        "=" * 55,
        f"  Stage 2 — Corner Detector  [{m['split']} split]",
        "=" * 55,
        f"  Total images     : {m['total']}",
        f"  Detected         : {m['detected']}  ({m['detection_rate_pct']:.1f}%)",
        f"  No detection     : {m['no_detection']}",
        f"  Mean dist error  : {m['mean_corner_dist_px']} px",
        f"  Median dist error: {m['median_corner_dist_px']} px",
        f"  P90 dist error   : {m['p90_dist_px']} px",
        f"  Within tolerance : {m['within_tolerance_pct']:.1f}%  ({m['tolerance']})",
    ]
    return "\n".join(lines)


# ── Stage 1 — Quality Gate classifier ─────────────────────────────────────────

def evaluate_stage1(split: str = "val", threshold: float = 0.5) -> dict:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

    model_path = _first_existing(STAGE1_MODEL, STAGE1_LEGACY_MODEL)
    if not model_path.is_file():
        print(f"Stage 1 model not found: {model_path}", file=sys.stderr)
        return {}

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, _NUM_STAGE1)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
    model.eval()

    transform = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_dir = STAGE1_DATA / "images" / split
    lbl_dir = STAGE1_DATA / "labels" / split
    run_dir = _make_run_dir("stage1")

    rows = []
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []

    prob_fields = [f"{c}_prob" for c in STAGE1_CLASSES]

    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue
        true_int = int(lbl_path.read_text().strip())
        if true_int >= _NUM_STAGE1:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]
        pred_int = int(probs.argmax().item())
        true_lbl = STAGE1_CLASSES[true_int]
        pred_lbl = STAGE1_CLASSES[pred_int]
        conf = float(probs[pred_int])
        correct = true_lbl == pred_lbl

        row: dict = {"stem": img_path.stem, "truth": true_lbl, "pred": pred_lbl, "confidence": round(conf, 4)}
        for i, field in enumerate(prob_fields):
            row[field] = round(float(probs[i]), 4)
        rows.append(row)
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)
        confidences.append(conf)
        correct_flags.append(correct)

        tag = f"T:{true_lbl[:8]} P:{pred_lbl[:8]} {conf:.2f}"
        thumb = _bordered(img, correct, tag)
        subfolder = "correct" if correct else "wrong"
        cv2.imwrite(str(run_dir / "viz" / subfolder / f"{img_path.stem}.jpg"), thumb)

    metrics = _classification_metrics(y_true, y_pred, STAGE1_CLASSES, "stage1", split, str(model_path))
    metrics["run_dir"] = str(run_dir)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "truth", "pred", "confidence"] + prob_fields)
        writer.writeheader()
        writer.writerows(rows)

    _save_confusion_matrix(y_true, y_pred, STAGE1_CLASSES, run_dir / "confusion_matrix.png")
    _save_confusion_matrix(
        [t for t in y_true if t in {"screen", "printout", "selfie", "back", "garbage"}],
        [p for t, p in zip(y_true, y_pred) if t in {"screen", "printout", "selfie", "back", "garbage"}],
        ("screen", "printout", "selfie", "back", "garbage"),
        run_dir / "confusion_matrix_rejects.png",
    )
    _save_confidence_hist(confidences, correct_flags, run_dir / "confidence_hist.png")

    report = _classification_report_text(metrics)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nOutputs: {run_dir}")
    return metrics


# ── Stage 3 — ID type classifier ─────────────────────────────────────────────

def evaluate_stage3(split: str = "val") -> dict:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    model_path = _first_existing(STAGE3_MODEL, STAGE3_LEGACY_MODEL)
    if not model_path.is_file():
        print(f"Stage 3 model not found: {model_path}", file=sys.stderr)
        return {}

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(STAGE3_CLASSES))
    model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
    model.eval()

    transform = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    split_dir = STAGE3_DATA / split
    run_dir = _make_run_dir("stage3")

    rows = []
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []
    class_samples: dict[str, list[np.ndarray]] = {c: [] for c in STAGE3_CLASSES}

    for cls_idx, cls in enumerate(STAGE3_CLASSES):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            pred_cls = STAGE3_CLASSES[pred_idx]
            conf = float(probs[pred_idx])
            correct = pred_cls == cls

            rows.append({"stem": img_path.stem, "truth": cls, "pred": pred_cls, "confidence": round(conf, 4)})
            y_true.append(cls)
            y_pred.append(pred_cls)
            confidences.append(conf)
            correct_flags.append(correct)

            # Collect class samples for grid (up to 6 per class, from val correct set)
            if correct and len(class_samples[cls]) < 6:
                class_samples[cls].append(img.copy())

            # Save viz thumbnails (cap at 200 wrong + 100 correct per stage to avoid OOM)
            wrong_count = sum(1 for f in (run_dir / "viz" / "wrong").iterdir()) if (run_dir / "viz" / "wrong").exists() else 0
            correct_count = sum(1 for f in (run_dir / "viz" / "correct").iterdir()) if (run_dir / "viz" / "correct").exists() else 0
            if (not correct and wrong_count < 200) or (correct and correct_count < 100):
                tag = f"TRUE:{cls[:8]} PRED:{pred_cls[:8]} {conf:.3f}"
                thumb = _bordered(img, correct, tag)
                subfolder = "correct" if correct else "wrong"
                cv2.imwrite(str(run_dir / "viz" / subfolder / f"{cls}_{img_path.stem}.jpg"), thumb)

    metrics = _classification_metrics(y_true, y_pred, STAGE3_CLASSES, "stage3", split, str(model_path))
    metrics["run_dir"] = str(run_dir)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w2 = csv.DictWriter(f, fieldnames=["stem", "truth", "pred", "confidence"])
        w2.writeheader()
        w2.writerows(rows)

    _save_confusion_matrix(y_true, y_pred, STAGE3_CLASSES, run_dir / "confusion_matrix.png")
    _save_confidence_hist(confidences, correct_flags, run_dir / "confidence_hist.png")
    _save_per_class_grid(class_samples, run_dir / "per_class_grid.png")

    report = _classification_report_text(metrics)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nOutputs: {run_dir}")
    return metrics


# ── Shared metrics ────────────────────────────────────────────────────────────

def _classification_metrics(
    y_true: list[str], y_pred: list[str],
    classes: tuple[str, ...], stage: str, split: str, model_path: str,
) -> dict:
    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / total if total else 0.0

    per_class = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    wrong_pairs = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    top_errors = [{"truth": t, "pred": p, "count": n}
                  for (t, p), n in wrong_pairs.most_common(10)]

    return {
        "stage": stage,
        "split": split,
        "model": model_path,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "top_errors": top_errors,
    }


def _classification_report_text(m: dict) -> str:
    lines = [
        "=" * 60,
        f"  {m['stage'].upper()} — {m['split']} split",
        "=" * 60,
        f"  Total images : {m['total']}",
        f"  Accuracy     : {m['accuracy']:.4f}  ({m['correct']}/{m['total']})",
        "",
        f"  {'Class':<22} {'Prec':>7} {'Recall':>7} {'F1':>7} {'n':>5}",
        f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*5}",
    ]
    for cls, s in m["per_class"].items():
        lines.append(f"  {cls:<22} {s['precision']:>7.4f} {s['recall']:>7.4f} {s['f1']:>7.4f} {s['support']:>5}")

    if m.get("top_errors"):
        lines += ["", "  Top misclassifications:"]
        for e in m["top_errors"]:
            lines.append(f"    {e['truth']:<20} -> {e['pred']:<20} x{e['count']}")

    if m.get("threshold_sweep"):
        lines += ["", f"  {'Thresh':>7} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8}"]
        for s in m["threshold_sweep"]:
            lines.append(f"  {s['t']:>7.2f} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f} {s['acc']:>8.4f}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pipeline models")
    parser.add_argument(
        "--stage",
        choices=[
            "stage1", "stage2", "stage3", "all",
            "quality_gate", "corners", "id_type",
            "legacy_stage1", "legacy_stage2", "legacy_stage4",
        ],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Data split to use (default: val)",
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Stage 1 classification threshold (default: 0.5)")
    parser.add_argument("--max-viz", type=int, default=20,
                        help="Max corner visualisation images for Stage 2 (default: 20)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save eval outputs (default: data/eval/ in repo). "
             "On Colab use Drive: /content/drive/MyDrive/id-forensics/eval",
    )
    args = parser.parse_args()

    if args.output_dir:
        set_eval_root(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Eval outputs → {args.output_dir}")

    ran_any = False

    stage = {
        "quality_gate": "stage1",
        "corners": "stage2",
        "id_type": "stage3",
        "legacy_stage2": "stage1",
        "legacy_stage1": "stage2",
        "legacy_stage4": "stage3",
    }.get(args.stage, args.stage)

    if stage in ("stage1", "all"):
        print("\n--- Stage 1: Quality gate ---")
        evaluate_stage1(split=args.split, threshold=args.threshold)
        ran_any = True

    if stage in ("stage2", "all"):
        print("\n--- Stage 2: Corner detection ---")
        evaluate_stage2(split=args.split, max_viz=args.max_viz)
        ran_any = True

    if stage in ("stage3", "all"):
        print("\n--- Stage 3: ID type classifier ---")
        evaluate_stage3(split=args.split)
        ran_any = True

    if not ran_any:
        print("No stage specified.", file=sys.stderr)
        return 1

    print(f"\nAll outputs under: {_EVAL_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
