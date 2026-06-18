# Sync Google Drive download -> local models/ (run after Colab training)
# Usage (from repo root in PowerShell):
#   .\scripts\sync_from_cloud.ps1 -Stage stage1
#   .\scripts\sync_from_cloud.ps1 -Stage stage2 -SourcePath "$env:USERPROFILE\Downloads\stage2_best.pt"

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("stage1", "stage2", "stage4")]
    [string]$Stage,

    [string]$SourcePath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$destMap = @{
    stage1 = "models\stage1_corners\weights\best.pt"
    stage2 = "models\stage2_screen\best.pt"
    stage4 = "models\stage4_id_type\best.pt"
}

$defaultNames = @{
    stage1 = "stage1_best.pt"
    stage2 = "stage2_best.pt"
    stage4 = "stage4_best.pt"
}

if (-not $SourcePath) {
    $SourcePath = Join-Path $env:USERPROFILE "Downloads\$($defaultNames[$Stage])"
}

$dest = Join-Path $Root $destMap[$Stage]

Write-Host "`n=== ID Forensics: sync FROM cloud ===" -ForegroundColor Cyan
Write-Host "  Source: $SourcePath"
Write-Host "  Dest:   $dest"

if (-not (Test-Path $SourcePath)) {
    Write-Host "`nFile not found. Download from Google Drive first:" -ForegroundColor Red
    Write-Host "  My Drive/id-forensics/outputs/$($defaultNames[$Stage])" -ForegroundColor Yellow
    Write-Host "  Save to Downloads, or pass -SourcePath 'C:\path\to\file.pt'" -ForegroundColor Yellow
    exit 1
}

$destDir = Split-Path -Parent $dest
if (-not (Test-Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
}

Copy-Item -Path $SourcePath -Destination $dest -Force
$mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
Write-Host "`nCopied ($mb MB). Evaluate with:" -ForegroundColor Green

if ($Stage -eq "stage1") {
    Write-Host "  .\venv\Scripts\python.exe scripts\evaluate_models.py --stage corners" -ForegroundColor White
} elseif ($Stage -eq "stage4") {
    Write-Host "  .\venv\Scripts\python.exe scripts\evaluate_models.py --stage id_type" -ForegroundColor White
} else {
    Write-Host "  .\venv\Scripts\python.exe scripts\evaluate_models.py --stage screen" -ForegroundColor White
}
Write-Host ""
