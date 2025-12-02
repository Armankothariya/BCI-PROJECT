# Complete cleanup and rebuild script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "COMPLETE SYSTEM CLEANUP & REBUILD" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

Set-Location "c:\Arman\Projects\BCI PROJECT"

Write-Host "`n[1/8] Deleting all models..." -ForegroundColor Cyan
Remove-Item -Path models\* -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[2/8] Deleting all results..." -ForegroundColor Cyan
Remove-Item -Path results\* -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[3/8] Deleting cache..." -ForegroundColor Cyan
Remove-Item -Path cache\* -Force -ErrorAction SilentlyContinue
Remove-Item -Path .streamlit\cache -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[4/8] Deleting logs..." -ForegroundColor Cyan
Remove-Item -Path logs\* -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[5/8] Deleting Python cache..." -ForegroundColor Cyan
Get-ChildItem -Path . -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[6/8] Deleting temporary files..." -ForegroundColor Cyan
Remove-Item -Path app.log -Force -ErrorAction SilentlyContinue
Remove-Item -Path app_state.db -Force -ErrorAction SilentlyContinue
Remove-Item -Path *.pyc -Force -ErrorAction SilentlyContinue
Remove-Item -Path fix_*.py -Force -ErrorAction SilentlyContinue
Remove-Item -Path quick_*.py -Force -ErrorAction SilentlyContinue
Remove-Item -Path check_*.py -Force -ErrorAction SilentlyContinue
Remove-Item -Path verify_*.py -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[7/8] Rebuilding data from pipeline..." -ForegroundColor Yellow
python run_pipeline.py
Write-Host "      Done" -ForegroundColor Green

Write-Host "[8/8] Verifying core files..." -ForegroundColor Cyan
if (Test-Path "app.py") { Write-Host "      OK: app.py" -ForegroundColor Green }
if (Test-Path "modules") { Write-Host "      OK: modules/" -ForegroundColor Green }
if (Test-Path "datasets") { Write-Host "      OK: datasets/" -ForegroundColor Green }
if (Test-Path "models/production_model.joblib") { Write-Host "      OK: production model" -ForegroundColor Green }

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext: Start the app with:" -ForegroundColor Yellow
Write-Host "  python -m streamlit run app.py" -ForegroundColor White
