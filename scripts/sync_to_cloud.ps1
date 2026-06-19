# Sync work PC -> GitHub
# Run this after: exporting fresh labels from Label Studio, or making code changes.
#
# Usage (from repo root):
#   .\scripts\sync_to_cloud.ps1
#   .\scripts\sync_to_cloud.ps1 -Message "add printout labels"

param(
    [string]$Message = "Update labels and code"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "`n=== ID Forensics: sync to GitHub ===" -ForegroundColor Cyan

# 1. Show what has changed
Write-Host "`n[1/2] Status" -ForegroundColor Yellow
git status --short

# 2. Stage label export + all code changes, then push
Write-Host "`n[2/2] Commit and push" -ForegroundColor Yellow

git add data/labels/label_studio_export.json
git add -u
git add notebooks/ scripts/ id_crop/ presentation_attack/ id_type/ orchestration/ tampering_detection/ field_extractor/ tests/

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "  Nothing to commit." -ForegroundColor DarkGray
    exit 0
}

git commit -m $Message
git push origin main

Write-Host "`nDone. Colab will get the latest on next git pull." -ForegroundColor Green
Write-Host @"

Next steps:
  - In Colab colab_00_setup.ipynb: run setup cell (sync_images=False to skip S3, True if new labels)
"@ -ForegroundColor White
