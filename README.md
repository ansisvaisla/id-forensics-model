# ID Forensics Model

Multi-stage forensic pipeline for ID document verification (FRAUD-1045).
Detects spoofed, replayed, injected, or tampered ID photos submitted by loan applicants.
Initial target: **Kenyan National IDs** (Legacy, Maisha, Huduma, Passport).

---

## Pipeline stages

| Stage | Module | Description | Status |
|-------|--------|-------------|--------|
| 1 | `id_crop/` | YOLOv8-OBB corner detection → `warpPerspective` deskew | Trained (mAP50 0.995) |
| 2 | `presentation_attack/` | Screen replay detection (EfficientNet-B0) | Trained (test F1 0.897) |
| 3 | `tampering_detection/` | ELA + EXIF metadata analysis | Algorithmic v1 |
| 4 | `id_type/` | ID type classification (Legacy / Maisha / Passport…) | Stub — needs more labels |
| 5 | `field_extractor/` | AWS Textract OCR + position-based field parser | Needs AWS credentials |
| 7 | `orchestration/` | Decision matrix, shadow-mode wrapper, risk tiers | Complete |

---

## Quick start

### Prerequisites

- Python 3.13
- CUDA GPU optional (CPU works; GPU ~5× faster for training)

### 1 — Clone and create virtual environment

```bash
git clone <repo-url>
cd id-forensics-model
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials and database connection string
```

`.env` variables needed:

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=       # only for SSO / temporary credentials
AWS_DEFAULT_REGION=af-south-1
S3_BUCKET=sf-zenka-ke-prod-media-svc

DB_HOST=
DB_PORT=5432
DB_NAME=
DB_USER=
DB_PASSWORD=
```

### 4 — Run the pipeline on one image

```python
from orchestration import run

with open("path/to/id.jpg", "rb") as f:
    result = run(f.read())

print(result.is_screen_replay, result.risk_tier)
```

---

## Model weights

Weights are **not committed** (large binaries). See [`models/README.md`](models/README.md) for
artifact locations and training commands.

| Model | Path | Size |
|-------|------|------|
| Stage 1 corners | `models/stage1_corners/weights/best.pt` | 6.3 MB |
| Stage 2 screen | `models/stage2_screen/best.pt` | 15.6 MB |

Download from the shared drive / S3 artifact bucket and place at the paths above.

---

## Batch labeling loop

`scripts/batch_label.py` automates the label → train cycle:

1. Queries postgres2 for recent ID-front images not yet in the label export
2. Runs the pipeline on each → generates Label Studio pre-annotations
3. Writes `data/batches/<timestamp>_batch.json` — import this into Label Studio

```powershell
# Generate a batch of 1000 pre-annotated tasks (looks back 30 days)
python scripts/batch_label.py --limit 1000 --hours 720

# Faster — no pipeline inference, fully manual labeling
python scripts/batch_label.py --limit 1000 --skip-inference
```

In Label Studio: **Import** → select the JSON file → skim predictions → correct wrong ones → **Export JSON** → save as `data/labels/label_studio_export.json`.

Then push and retrain:

```powershell
.\scripts\sync_to_cloud.ps1 -Message "batch labels"
# Colab workbench: SYNC_IMAGES=True, REBUILD_DATASET=True → retrain
```

**Pre-annotation quality mapping:**

| Pipeline output | LS `quality` label |
|---|---|
| `is_screen_replay` | `screen` |
| `is_printout` | `printout` |
| `crop.label == selfie_instead_of_document` | `selfie` |
| `crop.label == no_id_detected` | `garbage` |
| `is_partial_document` | `partial` |
| clean ID | `good_front` |

`id_type` is pre-filled from Stage 4 (legacy / maisha / huduma / passport / …).

If any model weights are missing the task is still imported — just without predictions.

---

## Training

### Stage 1 — Corner detector (YOLOv8-OBB)

```bash
python scripts/training/train_stage1_corners.py
```

Expects dataset at `data/yolo/corners/` (train/val/test splits).
To regenerate from Label Studio export:

```bash
python scripts/convert_labels_to_yolo.py
python scripts/split_yolo_dataset.py
```

### Stage 2 — Screen classifier (EfficientNet-B0)

```bash
python scripts/training/train_stage2_screen.py
# No GPU / behind a firewall? Use:
python scripts/training/train_stage2_screen.py --no-pretrained
```

Expects dataset at `data/yolo/screen/` (train/val/test splits, label 0=screen 1=live).

---

## Evaluation

```bash
# Both models
python scripts/evaluate_models.py

# One stage at a time
python scripts/evaluate_models.py --stage screen
python scripts/evaluate_models.py --stage corners
```

Outputs written to `data/eval/`:

| File | Contents |
|------|----------|
| `screen_metrics.json` | Precision, recall, F1, confusion matrix, threshold sweep |
| `screen_misclassified.csv` | Filenames + scores for wrong predictions |
| `screen_viz/` | Annotated thumbnails — green border = correct, red = wrong |
| `corners_metrics.json` | Mean distance error, % within 5% tolerance |
| `corners_viz/` | Overlay images (green = true corners, red = predicted) |

---

## Smoke test

```bash
python scripts/smoke_test_pipeline.py
# More images per category:
python scripts/smoke_test_pipeline.py --n 10
```

Runs `orchestration.run()` on known screen / good_front / garbage images and writes
`data/eval/smoke_report.json`.

---

## Tests

```bash
pytest tests/ -v
```

---

## Iterative improvement workflow

```
Label more images in Label Studio
        ↓
Export  →  data/labels/label_studio_export.json
        ↓
python scripts/convert_labels_to_yolo.py    # regenerate YOLO labels
python scripts/split_yolo_dataset.py         # rebuild splits
        ↓
python scripts/training/train_stage2_screen.py   # retrain
        ↓
python scripts/evaluate_models.py               # check test metrics
        ↓
Open data/eval/screen_viz/  in Explorer        # eyeball all 71 thumbnails
Check  data/eval/screen_misclassified.csv       # find patterns in errors
        ↓
Label the misclassified images correctly in Label Studio → repeat
```

**Where to find labeling candidates:**
- `data/eval/screen_misclassified.csv` — images the model got wrong → confirm/fix labels
- `data/eval/screen_viz/*_WRONG.jpg` — visual view of same
- Run the DB query for low `document_liveness_probability` to find more screen candidates

---

## Data transfer to home GPU (no S3/DB access needed)

Training only needs **code (GitHub)** + **YOLO dataset (zip)**. No AWS or DB credentials required.

### What goes where

| Item | GitHub | Zip transfer |
|------|--------|--------------|
| Python code, scripts, tests | yes | — |
| `data/labels/label_studio_export.json` | yes | also in zip |
| `data/yolo/` (images + labels + splits) | no — too large | **yes** |
| `data/raw/` (original images) | no | only if re-converting at home |
| `models/*.pt` (weights) | no | optional (`--include-models`) |
| `.env` (AWS/DB secrets) | never | not needed for training |

### On work PC (before leaving)

```powershell
cd id-forensics-model
.\venv\Scripts\activate

# Regenerate YOLO dataset from latest labels
python scripts/convert_labels_to_yolo.py
python scripts/split_yolo_dataset.py

# Create transfer zip (~hundreds of MB)
python scripts/pack_for_home.py
# Optional: include existing weights as starting point
python scripts/pack_for_home.py --include-models
```

Copy `id_forensics_home_data.zip` to USB / personal cloud.

### On home PC

```powershell
git clone <your-repo-url>
cd id-forensics-model

# Extract zip into repo root (creates data/yolo/ etc.)
Expand-Archive -Path ..\id_forensics_home_data.zip -DestinationPath . -Force

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# CUDA PyTorch (if pip installed CPU-only build):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python scripts/verify_home_setup.py
python scripts/training/train_stage2_screen.py
python scripts/evaluate_models.py
```

### Push code to GitHub (first time)

```powershell
# On work PC — after commits are ready
git remote add origin https://github.com/<your-org>/id-forensics-model.git
git push -u origin master
```

Create the empty repo on GitHub first (no README — you already have one locally).

AWS credentials are only needed for:
- Downloading new images from S3 (`scripts/download_from_manifest.py`)
- Field extraction (`field_extractor/`, Stage 5 — calls AWS Textract)

---

## Project layout

```
id-forensics-model/
├── id_crop/                  # Stage 1 — deskew & orientation
├── presentation_attack/      # Stage 2 — screen / printout detection
├── tampering_detection/      # Stage 3 — ELA, EXIF
├── id_type/                  # Stage 4 — ID type classifier (stub)
├── field_extractor/          # Stage 5 — Textract OCR
├── orchestration/            # Stage 7 — decision matrix
│   └── results.py            # shared dataclasses
├── scripts/
│   ├── training/             # train_stage1_corners.py, train_stage2_screen.py
│   ├── convert_labels_to_yolo.py
│   ├── split_yolo_dataset.py
│   ├── evaluate_models.py
│   ├── smoke_test_pipeline.py
│   └── download_from_manifest.py
├── tests/                    # pytest unit tests
├── docs/
│   └── jira_specifications.md
├── models/
│   └── README.md             # weight artifact registry
├── data/
│   └── labels/               # label_studio_export.json (committed)
├── requirements.txt
└── .env.example
```

---

## Spec

See [`docs/jira_specifications.md`](docs/jira_specifications.md) for full product specification (FRAUD-1045).
