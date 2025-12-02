# Safe Cleanup Script for BCI Project
# Removes only generated files, keeps core code and data

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "BCI PROJECT - SAFE CLEANUP SCRIPT" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Set location
Set-Location "c:\Arman\Projects\BCI PROJECT"

Write-Host "[1/7] Deleting models folder..." -ForegroundColor Cyan
Remove-Item -Path models\* -Force -ErrorAction SilentlyContinue
Write-Host "      ✓ Models deleted" -ForegroundColor Green

Write-Host "[2/7] Deleting results folder..." -ForegroundColor Cyan
Remove-Item -Path results\* -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      ✓ Results deleted" -ForegroundColor Green

Write-Host "[3/7] Deleting cache folder..." -ForegroundColor Cyan
Remove-Item -Path cache\* -Force -ErrorAction SilentlyContinue
Write-Host "      ✓ Cache deleted" -ForegroundColor Green

Write-Host "[4/7] Deleting logs folder..." -ForegroundColor Cyan
Remove-Item -Path logs\* -Force -ErrorAction SilentlyContinue
Write-Host "      ✓ Logs deleted" -ForegroundColor Green

Write-Host "[5/7] Deleting Python cache (__pycache__)..." -ForegroundColor Cyan
Get-ChildItem -Path . -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      ✓ Python cache deleted" -ForegroundColor Green

Write-Host "[6/7] Deleting temporary files..." -ForegroundColor Cyan
Remove-Item -Path app.log -Force -ErrorAction SilentlyContinue
Remove-Item -Path app_state.db -Force -ErrorAction SilentlyContinue
Remove-Item -Path confusion_matrix.png -Force -ErrorAction SilentlyContinue
Remove-Item -Path check_*.py -Force -ErrorAction SilentlyContinue
Write-Host "      ✓ Temporary files deleted" -ForegroundColor Green

Write-Host "[7/7] Verifying core files are intact..." -ForegroundColor Cyan
$coreFiles = @("app.py", "config.yaml", "fix_overfitting.py", "run_pipeline.py", "modules", "datasets", "music")
$allPresent = $true
foreach ($file in $coreFiles) {
    if (Test-Path $file) {
        Write-Host "      ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "      ✗ $file MISSING!" -ForegroundColor Red
        $allPresent = $false
    }
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

if ($allPresent) {
    Write-Host "✓ CLEANUP COMPLETE - All core files intact!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Run: python fix_overfitting.py" -ForegroundColor White
    Write-Host "2. Run: move models\production_model_fixed.joblib models\production_model.joblib" -ForegroundColor White
    Write-Host "3. Run: streamlit run app.py" -ForegroundColor White
} else {
    Write-Host "⚠ WARNING: Some core files are missing!" -ForegroundColor Red
}

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
