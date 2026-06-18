# Sync work PC -> GitHub + Google Drive (run before Colab training)
# Usage (from repo root in PowerShell):
#   .\scripts\sync_to_cloud.ps1
#   .\scripts\sync_to_cloud.ps1 -SkipPack
#   .\scripts\sync_to_cloud.ps1 -GitPush

param(
    [switch]$SkipPack,
    [switch]$GitPush,
    [string]$CommitMessage = "Update labels and training data for Colab"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "`n=== ID Forensics: sync TO cloud ===" -ForegroundColor Cyan

if (-not $SkipPack) {
    Write-Host "`n[1/3] Convert labels + split + pack..." -ForegroundColor Yellow
    & .\venv\Scripts\python.exe scripts\convert_labels_to_yolo.py
    & .\venv\Scripts\python.exe scripts\split_yolo_dataset.py
    & .\venv\Scripts\python.exe scripts\split_id_type_dataset.py
    # Pack with stage1 model included — Colab needs it to deskew id_type images before stage4 training
    & .\venv\Scripts\python.exe scripts\pack_for_home.py --include-stage1
    $zip = Join-Path $Root "id_forensics_home_data.zip"
    if (Test-Path $zip) {
        $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
        Write-Host "  Created: $zip ($mb MB)" -ForegroundColor Green
    }
} else {
    Write-Host "`n[1/3] Skipped pack (-SkipPack)" -ForegroundColor DarkGray
}

Write-Host "`n[2/3] Upload zip to Google Drive" -ForegroundColor Yellow
Write-Host @"
  1. Open https://drive.google.com
  2. Create folder: id-forensics  (if missing)
  3. Upload (overwrite): id_forensics_home_data.zip
     -> My Drive/id-forensics/id_forensics_home_data.zip
"@ -ForegroundColor White

Write-Host "`n[3/3] Push code to GitHub" -ForegroundColor Yellow
git status
if ($GitPush) {
    git add -u
    git add notebooks/ scripts/colab_bootstrap.py scripts/sync_to_cloud.ps1 scripts/sync_from_cloud.ps1
    git commit -m $CommitMessage
    git push origin main
    Write-Host "  Pushed to origin/main" -ForegroundColor Green
} else {
    Write-Host @"
  Copy/paste if you have code changes:
    git add -u
    git add notebooks/ scripts/colab_bootstrap.py scripts/sync_*.ps1
    git commit -m "your message"
    git push origin main
"@ -ForegroundColor White
}

Write-Host "`nDone. Open VS Code -> colab_00_setup.ipynb -> run all cells -> train notebook.`n" -ForegroundColor Cyan
