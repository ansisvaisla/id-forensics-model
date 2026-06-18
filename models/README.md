# Model Artifacts

All model weights are gitignored. This file documents each artifact.

## Stage 1 — ID Crop: Corner Detector

| Key | Value |
|---|---|
| Path | `models/stage1_corners/weights/best.pt` |
| Architecture | YOLOv8n-OBB (Oriented Bounding Box) |
| Task | Detect 4 corners of Kenyan ID card |
| Training data | `data/yolo/corners/` — 361 images, 253 train / 54 val / 54 test |
| Labels source | Label Studio polygon export (`data/labels/label_studio_export.json`) |
| Training script | `scripts/training/train_stage1_corners.py` |
| Output format | `0 x1 y1 x2 y2 x3 y3 x4 y4` (normalised 0–1) |
| Status | Not yet trained |

**To train:**
```powershell
cd scripts\training
..\..\venv\Scripts\python.exe train_stage1_corners.py --epochs 50 --device cpu
```

---

## Stage 2 — Presentation Attack: Screen Detector

| Key | Value |
|---|---|
| Path | `models/stage2_screen/best.pt` |
| Architecture | EfficientNet-B0 (pretrained ImageNet, fine-tuned) |
| Task | Binary classification: screen replay (0) vs live genuine (1) |
| Training data | `data/yolo/screen/` — 472 images, 330 train / 71 val / 71 test |
| Class balance | 103 screen positives vs 369 live negatives (weighted loss) |
| Training script | `scripts/training/train_stage2_screen.py` |
| Status | Not yet trained |

**To train:**
```powershell
cd scripts\training
..\..\venv\Scripts\python.exe train_stage2_screen.py --epochs 30 --device cpu
```

---

## Stage 3 — Tampering Detection

No model artifact. Fully algorithmic (ELA + EXIF heuristics).
See `tampering_detection/__init__.py` for thresholds.

---

## Stage 4 — ID Type Classifier

| Key | Value |
|---|---|
| Path | `models/stage4_id_type/best.pt` |
| Architecture | EfficientNet-B0 |
| Task | 7-class: legacy / maisha / huduma / passport / driving_licence / foreign_document / unknown_id |
| Status | Deferred — insufficient huduma/passport labels. v1 returns 'unknown'. |

---

## Stage 5 — Field Extractor

No model artifact. Uses AWS Textract + position-based heuristics.
See `field_extractor/__init__.py`.

---

## Logging

Training runs should be logged to MLflow or W&B.
Never print raw metrics to stdout in production code.
Set `MLFLOW_TRACKING_URI` or `WANDB_API_KEY` in `.env`.
