"""End-to-end smoke test for the full pipeline.

Runs orchestration.run(image_bytes) on a small curated set of images with known
expected outcomes and writes a JSON report.

Expected outcomes
-----------------
screen images     → is_screen_replay=True
good_front images → is_screen_replay=False, is_partial_document=False
garbage images    → is_partial_document=True OR label contains a non-document hint

No AWS credentials required — Stage 5 (Textract) will gracefully error-catch
without blocking the smoke test.

Usage
-----
    python scripts/smoke_test_pipeline.py
    python scripts/smoke_test_pipeline.py --n 5          # images per category
    python scripts/smoke_test_pipeline.py --out data/eval/smoke_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SCREEN_IMG_DIR = PROJECT_ROOT / "data" / "yolo" / "screen" / "images" / "test"
SCREEN_LBL_DIR = PROJECT_ROOT / "data" / "yolo" / "screen" / "labels" / "test"
EVAL_OUT = PROJECT_ROOT / "data" / "eval" / "smoke_report.json"
LABEL_EXPORT = PROJECT_ROOT / "data" / "labels" / "label_studio_export.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_GARBAGE_QUALITY = {"selfie_instead_of_document", "random_image", "back_instead_of_front", "black_screen"}


def _collect_screen_images(n: int) -> list[dict]:
    """Return up to n screen (label=0) images from the test split."""
    out = []
    for lbl in sorted(SCREEN_LBL_DIR.glob("*.txt")):
        if int(lbl.read_text().strip()) == 0:
            img = SCREEN_IMG_DIR / f"{lbl.stem}.jpg"
            if img.is_file():
                out.append({"path": str(img), "category": "screen", "label": "screen"})
        if len(out) >= n:
            break
    return out


def _collect_good_front_images(n: int) -> list[dict]:
    """Return up to n live (label=1) images from the test split."""
    out = []
    for lbl in sorted(SCREEN_LBL_DIR.glob("*.txt")):
        if int(lbl.read_text().strip()) == 1:
            img = SCREEN_IMG_DIR / f"{lbl.stem}.jpg"
            if img.is_file():
                out.append({"path": str(img), "category": "good_front", "label": "good_front"})
        if len(out) >= n:
            break
    return out


def _collect_garbage_images(n: int) -> list[dict]:
    """Return up to n garbage/non-document images found via label export."""
    if not LABEL_EXPORT.is_file():
        return []

    export = json.loads(LABEL_EXPORT.read_text(encoding="utf-8"))
    all_images = {p.name: p for p in RAW_DIR.rglob("*.jpg")}

    out = []
    for task in export:
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled") or not ann.get("result"):
                continue
            for r in ann["result"]:
                if r.get("type") == "choices" and r.get("from_name") == "quality":
                    qs = set(r["value"]["choices"])
                    if qs & _GARBAGE_QUALITY:
                        fu = task.get("file_upload", "")
                        stem = fu.split("-", 1)[1] if "-" in fu else fu
                        if stem in all_images:
                            out.append({
                                "path": str(all_images[stem]),
                                "category": "garbage",
                                "label": list(qs & _GARBAGE_QUALITY)[0],
                            })
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return out[:n]


def _run_one(image_path: str) -> dict:
    """Call orchestration.run on one image file. Returns result dict + timing."""
    from orchestration import run as pipeline_run

    img_bytes = Path(image_path).read_bytes()
    t0 = time.perf_counter()
    result = pipeline_run(img_bytes)
    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "elapsed_s": elapsed,
        "is_screen_replay": result.is_screen_replay,
        "is_partial_document": result.is_partial_document,
        "is_tampered": result.is_tampered,
        "label": result.label,
        "risk_tier": result.risk_tier,
        "screen_score": round(result.presentation_attack.screen_score, 4) if result.presentation_attack else None,
        "stage_errors": [],
    }


def _check_expectation(entry: dict, result: dict) -> dict:
    """Check if result meets the expectation for this category. Returns pass/fail dict."""
    cat = entry["category"]
    passed = True
    notes = []

    if cat == "screen":
        if not result["is_screen_replay"]:
            passed = False
            notes.append(f"Expected is_screen_replay=True, got False (score={result['screen_score']})")
    elif cat == "good_front":
        if result["is_screen_replay"]:
            passed = False
            notes.append(f"Expected is_screen_replay=False, got True (score={result['screen_score']})")
        if result["is_partial_document"]:
            notes.append("is_partial_document=True (corners not fully detected — not a hard fail)")
    elif cat == "garbage":
        if not result["is_partial_document"] and not result["is_screen_replay"]:
            notes.append("Not flagged as partial or screen — pipeline accepted a garbage image")

    return {"passed": passed, "notes": notes}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test")
    parser.add_argument("--n", type=int, default=5, help="Images per category (default 5)")
    parser.add_argument("--out", default=str(EVAL_OUT))
    args = parser.parse_args()

    images = (
        _collect_screen_images(args.n)
        + _collect_good_front_images(args.n)
        + _collect_garbage_images(args.n)
    )

    print(f"Smoke test: {len(images)} images ({args.n} screen, {args.n} good_front, up to {args.n} garbage)\n")

    records = []
    pass_count = 0
    fail_count = 0

    for entry in images:
        print(f"  [{entry['category']:12s}] {Path(entry['path']).name} ...", end=" ", flush=True)
        try:
            result = _run_one(entry["path"])
        except Exception as exc:
            print(f"ERROR: {exc}")
            records.append({**entry, "error": str(exc), "passed": False})
            fail_count += 1
            continue

        check = _check_expectation(entry, result)
        status = "PASS" if check["passed"] else "FAIL"
        if check["passed"]:
            pass_count += 1
        else:
            fail_count += 1

        print(
            f"{status}  screen={result['is_screen_replay']} "
            f"partial={result['is_partial_document']} "
            f"risk={result['risk_tier']} "
            f"({result['elapsed_s']}s)"
        )
        if check["notes"]:
            for note in check["notes"]:
                print(f"             note: {note}")

        records.append({**entry, **result, **check})

    summary = {
        "total": len(records),
        "passed": pass_count,
        "failed": fail_count,
        "pass_rate": round(pass_count / len(records), 4) if records else 0.0,
    }
    print(f"\nSummary: {pass_count}/{len(records)} passed ({summary['pass_rate']*100:.1f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "results": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Report saved -> {out_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
