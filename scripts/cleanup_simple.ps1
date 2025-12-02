# Safe Cleanup Script for BCI Project
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "BCI PROJECT - SAFE CLEANUP" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "c:\Arman\Projects\BCI PROJECT"

Write-Host "[1/7] Deleting models folder..." -ForegroundColor Cyan
Remove-Item -Path "models\*" -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[2/7] Deleting results folder..." -ForegroundColor Cyan
Remove-Item -Path "results\*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[3/7] Deleting cache folder..." -ForegroundColor Cyan
Remove-Item -Path "cache\*" -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[4/7] Deleting logs folder..." -ForegroundColor Cyan
Remove-Item -Path "logs\*" -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[5/7] Deleting Python cache..." -ForegroundColor Cyan
Get-ChildItem -Path . -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[6/7] Deleting temporary files..." -ForegroundColor Cyan
Remove-Item -Path "app.log" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "app_state.db" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "confusion_matrix.png" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "check_*.py" -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[7/7] Verifying core files..." -ForegroundColor Cyan
if (Test-Path "app.py") { Write-Host "      OK: app.py" -ForegroundColor Green }
if (Test-Path "config.yaml") { Write-Host "      OK: config.yaml" -ForegroundColor Green }
if (Test-Path "modules") { Write-Host "      OK: modules/" -ForegroundColor Green }
if (Test-Path "datasets") { Write-Host "      OK: datasets/" -ForegroundColor Green }
if (Test-Path "music") { Write-Host "      OK: music/" -ForegroundColor Green }

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. python fix_overfitting.py" -ForegroundColor White
Write-Host "2. move models\production_model_fixed.joblib models\production_model.joblib" -ForegroundColor White
Write-Host "3. streamlit run app.py" -ForegroundColor White
