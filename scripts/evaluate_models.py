"""Evaluate trained Stage 1 (corners) and Stage 2 (screen) models on held-out test sets.

Outputs:
    data/eval/screen_metrics.json   — precision, recall, F1, confusion matrix, threshold sweep
    data/eval/screen_misclassified.csv — filenames + scores for wrong predictions
    data/eval/corners_metrics.json  — mean corner distance error, % within tolerance
    data/eval/corners_viz/          — overlay images (predicted vs true corners, up to 10)

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --stage screen
    python scripts/evaluate_models.py --stage corners
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREEN_DATA = PROJECT_ROOT / "data" / "yolo" / "screen"
CORNERS_DATA = PROJECT_ROOT / "data" / "yolo" / "corners"
SCREEN_MODEL = PROJECT_ROOT / "models" / "stage2_screen" / "best.pt"
CORNERS_MODEL = PROJECT_ROOT / "models" / "stage1_corners" / "weights" / "best.pt"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"


# ---------------------------------------------------------------------------
# Stage 2 — Screen classifier evaluation
# ---------------------------------------------------------------------------

def _save_screen_viz(results: list[dict], img_dir: "Path", threshold: float) -> None:
    """Save annotated thumbnails for every test image to data/eval/screen_viz/.

    Each image gets a border + label overlay so you can eyeball them in Explorer:
      GREEN border  = correct prediction
      RED border    = wrong prediction (misclassified)
    Title shows: true label | predicted label | screen_prob score
    """
    import cv2
    import numpy as np

    viz_dir = EVAL_DIR / "screen_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    THUMB = 300
    LABEL_MAP = {0: "SCREEN", 1: "LIVE"}
    BORDER = 6

    for r in results:
        img_path = img_dir / f"{r['stem']}.jpg"
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        scale = THUMB / max(h, w)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)))

        correct = r["true"] == r["pred"]
        color = (0, 180, 0) if correct else (0, 0, 220)

        # Coloured border
        th, tw = thumb.shape[:2]
        canvas = np.full((th + 2 * BORDER + 36, tw + 2 * BORDER, 3), 30, dtype=np.uint8)
        canvas[BORDER: BORDER + th, BORDER: BORDER + tw] = thumb
        cv2.rectangle(canvas, (0, 0), (tw + 2 * BORDER - 1, th + 2 * BORDER + 35), color, BORDER)

        label_text = (
            f"TRUE:{LABEL_MAP[r['true']]} PRED:{LABEL_MAP[r['pred']]} "
            f"p={r['screen_prob']:.3f}"
        )
        cv2.putText(canvas, label_text, (BORDER, th + 2 * BORDER + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

        suffix = "OK" if correct else "WRONG"
        cv2.imwrite(str(viz_dir / f"{r['stem']}_{suffix}.jpg"), canvas)


def evaluate_screen(threshold: float = 0.5) -> dict:
    """Run Stage 2 screen model on test split. Returns metrics dict."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    import cv2
    import numpy as np

    if not SCREEN_MODEL.is_file():
        print(f"Screen model not found: {SCREEN_MODEL}", file=sys.stderr)
        return {}

    # Load model
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(str(SCREEN_MODEL), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_dir = SCREEN_DATA / "images" / "test"
    lbl_dir = SCREEN_DATA / "labels" / "test"

    results = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue
        true_label = int(lbl_path.read_text(encoding="utf-8").strip())
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]
        screen_prob = float(probs[0])
        pred_label = 0 if screen_prob >= threshold else 1
        results.append({
            "stem": img_path.stem,
            "true": true_label,
            "pred": pred_label,
            "screen_prob": screen_prob,
        })

    # Metrics per threshold sweep
    def _metrics_at(thresh: float) -> dict:
        tp = fp = tn = fn = 0
        for r in results:
            p = 0 if r["screen_prob"] >= thresh else 1
            if p == 0 and r["true"] == 0: tp += 1
            elif p == 0 and r["true"] == 1: fp += 1
            elif p == 1 and r["true"] == 1: tn += 1
            else: fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        acc = (tp + tn) / len(results) if results else 0.0
        return {
            "threshold": thresh,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    primary = _metrics_at(threshold)
    sweep = [_metrics_at(round(t, 2)) for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]]

    misclassified = [r for r in results if r["true"] != r["pred"]]

    # Save misclassified CSV
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    mis_path = EVAL_DIR / "screen_misclassified.csv"
    with open(mis_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "true", "pred", "screen_prob"])
        w.writeheader()
        w.writerows(misclassified)

    # --- Save visualisation strips: correct predictions + misclassified ---
    _save_screen_viz(results, img_dir, threshold)

    n_screen = sum(1 for r in results if r["true"] == 0)
    n_live = sum(1 for r in results if r["true"] == 1)
    metrics = {
        "model": str(SCREEN_MODEL),
        "test_total": len(results),
        "test_screen_positives": n_screen,
        "test_live_negatives": n_live,
        "primary_threshold": threshold,
        "metrics": primary,
        "threshold_sweep": sweep,
        "misclassified_count": len(misclassified),
        "misclassified_file": str(mis_path),
    }
    out = EVAL_DIR / "screen_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Stage 1 — Corners evaluation
# ---------------------------------------------------------------------------


def evaluate_corners(tolerance_pct: float = 0.05, max_viz: int = 10) -> dict:
    """Run Stage 1 corner model on test split. Returns metrics dict.

    Supports both Pose (keypoints) and OBB (legacy) model weights.
    Viz draws matched pairs as lines: GREEN=true corner, RED=predicted, YELLOW=connecting line.
    """
    import cv2
    import numpy as np
    import math
    from itertools import permutations

    if not CORNERS_MODEL.is_file():
        print(f"Corners model not found: {CORNERS_MODEL}", file=sys.stderr)
        return {}

    from ultralytics import YOLO
    model = YOLO(str(CORNERS_MODEL))

    img_dir = CORNERS_DATA / "images" / "test"
    lbl_dir = CORNERS_DATA / "labels" / "test"
    viz_dir = EVAL_DIR / "corners_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    distances = []
    within_tol = 0
    no_detection = 0
    viz_count = 0
    per_image = []

    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue

        parts = lbl_path.read_text().strip().split()
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Parse ground-truth corners from label file.
        # Pose format:   0 cx cy bw bh  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4  (17 values)
        # OBB format:    0 x1 y1 x2 y2 x3 y3 x4 y4  (9 values)
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 16:
            # Pose: skip bbox (first 4), then take x y (skip visibility) for each of 4 kpts
            true_pts = np.array([
                [vals[4 + i * 3] * w, vals[5 + i * 3] * h]
                for i in range(4)
            ], dtype=np.float32)
        elif len(vals) == 8:
            # OBB legacy
            true_pts = np.array([
                [vals[i * 2] * w, vals[i * 2 + 1] * h]
                for i in range(4)
            ], dtype=np.float32)
        else:
            continue

        results = model(img, verbose=False)

        # Extract predicted corners (Pose or OBB)
        pred_pts = None
        if results:
            r = results[0]
            if (
                hasattr(r, "keypoints") and r.keypoints is not None
                and len(r.keypoints) > 0
            ):
                kpts = r.keypoints[0].xy
                if kpts is not None and len(kpts) > 0:
                    pred_pts = kpts[0].cpu().numpy()
            elif hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                pred_pts = r.obb[0].xyxyxyxy[0].cpu().numpy().reshape(4, 2)

        if pred_pts is None or len(pred_pts) < 4:
            no_detection += 1
            per_image.append({"stem": img_path.stem, "detected": False, "dist": None})
            continue

        # Optimal corner matching (permutations)
        best_dist = float("inf")
        best_perm = list(range(4))
        for perm in permutations(range(4)):
            d = sum(
                math.hypot(pred_pts[perm[i], 0] - true_pts[i, 0],
                           pred_pts[perm[i], 1] - true_pts[i, 1])
                for i in range(4)
            ) / 4
            if d < best_dist:
                best_dist = d
                best_perm = list(perm)

        distances.append(best_dist)
        diag = math.hypot(w, h)
        ok = best_dist < tolerance_pct * diag
        if ok:
            within_tol += 1

        per_image.append({
            "stem": img_path.stem,
            "detected": True,
            "dist": round(best_dist, 2),
            "within_tol": ok,
        })

        # Visualisation with matched pairs + connecting lines
        if viz_count < max_viz:
            vis = img.copy()
            matched_pred = pred_pts[best_perm]
            for i in range(4):
                tx, ty = int(true_pts[i, 0]), int(true_pts[i, 1])
                px, py = int(matched_pred[i, 0]), int(matched_pred[i, 1])
                cv2.line(vis, (tx, ty), (px, py), (0, 255, 255), 2)
                cv2.circle(vis, (tx, ty), 7, (0, 255, 0), -1)   # green = true
                cv2.circle(vis, (px, py), 7, (0, 0, 255), -1)   # red = pred
                err = math.hypot(px - tx, py - ty)
                cv2.putText(vis, f"{err:.0f}", (tx + 5, ty - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            label = "OK" if ok else "BAD"
            cv2.putText(vis, f"mean={best_dist:.1f}px {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(str(viz_dir / f"{img_path.stem}_eval.jpg"), vis)
            viz_count += 1

    n = len(distances)
    mean_dist = sum(distances) / n if n else 0.0
    pct_within = within_tol / (n + no_detection) * 100 if (n + no_detection) else 0.0

    metrics = {
        "model": str(CORNERS_MODEL),
        "test_total": n + no_detection,
        "detected": n,
        "no_detection": no_detection,
        "mean_corner_distance_px": round(mean_dist, 2),
        "within_tolerance_pct": round(pct_within, 2),
        "tolerance": f"{int(tolerance_pct * 100)}% of diagonal",
        "viz_dir": str(viz_dir),
        "per_image": per_image,
    }
    out = EVAL_DIR / "corners_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_screen(m: dict) -> None:
    if not m:
        return
    p = m["metrics"]
    print(f"\n=== Stage 2 — Screen Classifier ===")
    print(f"  Test set   : {m['test_total']} images  ({m['test_screen_positives']} screen, {m['test_live_negatives']} live)")
    print(f"  Threshold  : {m['primary_threshold']}")
    print(f"  Accuracy   : {p['accuracy']:.4f}")
    print(f"  Precision  : {p['precision']:.4f}  (of flagged-screen, how many really screen)")
    print(f"  Recall     : {p['recall']:.4f}  (of real screens, how many caught)")
    print(f"  F1         : {p['f1']:.4f}")
    print(f"  Confusion  : TP={p['tp']} FP={p['fp']} TN={p['tn']} FN={p['fn']}")
    print(f"  Misclassified: {m['misclassified_count']} -> {m['misclassified_file']}")
    print(f"\n  Threshold sweep:")
    print(f"  {'thresh':>7}  {'prec':>7}  {'recall':>7}  {'F1':>7}  {'acc':>7}")
    for s in m["threshold_sweep"]:
        print(f"  {s['threshold']:>7.2f}  {s['precision']:>7.4f}  {s['recall']:>7.4f}  {s['f1']:>7.4f}  {s['accuracy']:>7.4f}")


def _print_corners(m: dict) -> None:
    if not m:
        return
    print(f"\n=== Stage 1 — Corner Detector ===")
    print(f"  Test set       : {m['test_total']} images")
    print(f"  Detected       : {m['detected']}  (no detection: {m['no_detection']})")
    print(f"  Mean dist error: {m['mean_corner_distance_px']} px")
    print(f"  Within {m['tolerance']}: {m['within_tolerance_pct']:.1f}%")
    print(f"  Viz saved      : {m['viz_dir']}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained pipeline models")
    parser.add_argument("--stage", choices=["screen", "corners", "both"], default="both")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.stage in ("screen", "both"):
        print("Running Stage 2 screen evaluation...")
        m = evaluate_screen(threshold=args.threshold)
        _print_screen(m)

    if args.stage in ("corners", "both"):
        print("\nRunning Stage 1 corners evaluation...")
        m = evaluate_corners()
        _print_corners(m)

    print(f"\nFull results saved to: {EVAL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
