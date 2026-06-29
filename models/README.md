# Model Artifacts

All model weights are gitignored. This file documents each artifact.

## Stage 1 — Quality Gate

| Key | Value |
|---|---|
| Path | `models/stage1_quality_gate/best.pt` |
| Legacy path | `models/stage2_screen/best.pt` |
| Architecture | EfficientNet-B0 |
| Task | 8-class: screen / printout / selfie / back / garbage / good_front / partial / blurry |
| Training data | `data/yolo/screen/` |
| Training script | `scripts/training/train_stage2_screen.py` |
| Status | Trained |

**To train:**
```powershell
cd scripts\training
..\..\venv\Scripts\python.exe train_stage2_screen.py --epochs 60 --device cuda
```

---

## Stage 2 — ID Crop: Corner Detector

| Key | Value |
|---|---|
| Path | `models/stage2_corners/weights/best.pt` |
| Legacy path | `models/stage1_corners/weights/best.pt` |
| Architecture | YOLOv8n-Pose |
| Task | Detect 4 corners of Kenyan ID card |
| Training data | `data/yolo/corners/` |
| Labels source | Label Studio polygon export (`data/labels/label_studio_export.json`) |
| Training script | `scripts/training/train_stage1_corners.py` |
| Output format | `0 cx cy w h x1 y1 2 x2 y2 2 x3 y3 2 x4 y4 2` |
| Status | Trained |

**To train:**
```powershell
cd scripts\training
..\..\venv\Scripts\python.exe train_stage1_corners.py --epochs 50 --device cuda
```

---

## Stage 3 — ID Type Classifier

| Key | Value |
|---|---|
| Path | `models/stage3_id_type/best.pt` |
| Legacy path | `models/stage4_id_type/best.pt` |
| Architecture | EfficientNet-B0 |
| Task | 7-class: legacy / maisha / huduma / passport / driving_licence / foreign_document / unknown_id |
| Training script | `scripts/training/train_stage4_id_type.py` |
| Status | Trained |

---

## Stage 4 — Field Localization And OCR

No model artifact. Uses normalized field templates plus OCR word boxes.
See `field_localization/__init__.py`.

---

## Experimental Side Check — Tampering Detection

No model artifact. Fully algorithmic (ELA + EXIF heuristics).
See `tampering_detection/__init__.py` for thresholds.

---

## Logging

Training runs should be logged to MLflow or W&B.
Never print raw metrics to stdout in production code.
Set `MLFLOW_TRACKING_URI` or `WANDB_API_KEY` in `.env`.
