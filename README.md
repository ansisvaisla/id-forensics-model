# ID Forensics Model

Multi-stage forensic pipeline for ID document verification (FRAUD-1045).
Detects spoofed, replayed, injected, or tampered ID photos submitted by loan applicants.
Initial target: **Kenyan National IDs** (Legacy, Maisha, Huduma, Passport).

---

## Pipeline stages

| Stage | Module | Description | Status |
|-------|--------|-------------|--------|
| 1 | `quality_gate/` | 8-class raw-image quality gate: screen / printout / selfie / back / garbage / good_front / partial / blurry | Trained |
| 2 | `id_crop/` | YOLOv8-Pose corner detection → crop / optional `warpPerspective` deskew | Trained |
| 3 | `id_type/` | ID type classification (Legacy / Maisha / Passport…) | Trained |
| 4 | `field_localization/` | Layout-aware OCR field localization and parsing | Template v1 |
| Side check | `tampering_detection/` | ELA + EXIF metadata analysis, experimental only | Algorithmic v1 |
| Decision | `orchestration/` | Shadow-mode wrapper, decision matrix, risk tiers | Complete |

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
| Stage 1 quality gate | `models/stage1_quality_gate/best.pt` | 15.6 MB |
| Stage 2 corners | `models/stage2_corners/weights/best.pt` | 6.3 MB |
| Stage 3 ID type | `models/stage3_id_type/best.pt` | 15.6 MB |

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

`id_type` is pre-filled from Stage 3 (legacy / maisha / huduma / passport / …).

If any model weights are missing the task is still imported — just without predictions.

---

## Training

### Stage 1 — Quality gate (EfficientNet-B0)

```bash
python scripts/training/train_stage2_screen.py
```

Expects dataset at `data/yolo/screen/` (train/val/test splits).
To regenerate from Label Studio export:

```bash
python scripts/convert_labels_ls_to_quality_gate.py
```

### Stage 2 — Corner detector (YOLOv8-Pose)

```bash
python scripts/training/train_stage1_corners.py
```

Expects dataset at `data/yolo/corners/` (train/val/test splits).

### Stage 3 — ID type classifier (EfficientNet-B0)

```bash
python scripts/training/train_stage4_id_type.py
```

Expects dataset at `data/id_type/` (ImageFolder train/val/test splits).

### Stage 4 — Field localization and OCR

Stage 4 is template-based for now. It can audit existing AWS Rekognition OCR logs
without making new OCR calls:

```bash
python scripts/export_ocr_audit.py --limit 500 --out data/eval/ocr_audit.csv
```

After reviewing the CSV, add columns like `expected_id_number`,
`expected_name`, and `expected_date_of_birth`, then evaluate:

```bash
python scripts/evaluate_field_localization.py --input data/eval/ocr_audit_reviewed.csv
```

---

## Evaluation

```bash
# Both models
python scripts/evaluate_models.py

# One stage at a time
python scripts/evaluate_models.py --stage stage1  # quality gate
python scripts/evaluate_models.py --stage stage2  # corners
python scripts/evaluate_models.py --stage stage3  # ID type
```

Outputs written to `data/eval/`:

| File | Contents |
|------|----------|
| `data/eval/stage1/<run>/` | Quality gate metrics, confusion matrix, confidence histogram |
| `data/eval/stage2/<run>/` | Corner distance metrics and overlay thumbnails |
| `data/eval/stage3/<run>/` | ID type metrics, confusion matrix, confidence histogram |

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
python scripts/convert_labels_ls_to_quality_gate.py  # Stage 1 dataset
python scripts/convert_labels_to_yolo.py             # Stage 2/3 datasets
python scripts/split_yolo_dataset.py                 # rebuild splits
        ↓
python scripts/training/train_stage2_screen.py   # retrain Stage 1 quality gate
        ↓
python scripts/evaluate_models.py               # check test metrics
        ↓
Open data/eval/stage1/<latest>/viz/wrong/       # inspect quality gate errors
        ↓
Label the misclassified images correctly in Label Studio → repeat
```

**Where to find labeling candidates:**
- `data/eval/stage1/<latest>/predictions.csv` — images the quality gate got wrong
- `data/eval/stage1/<latest>/viz/wrong/` — visual view of same
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
- Field localization/OCR (`field_localization/`, Stage 4; local OCR optional)

---

## Project layout

```
id-forensics-model/
├── quality_gate/             # Stage 1 — raw image quality / attack gate
├── id_crop/                  # Stage 2 — corners, crop, optional deskew
├── id_type/                  # Stage 3 — ID type classifier
├── field_localization/       # Stage 4 — template OCR field extraction
├── field_extractor/          # compatibility wrapper for old imports
├── tampering_detection/      # experimental side check — ELA, EXIF
├── orchestration/            # decision matrix
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
