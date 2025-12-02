# Clear all cache, models, and results for fresh pipeline run

Write-Host "Clearing all cache, models, and results..." -ForegroundColor Cyan

# Remove cache files
if (Test-Path "cache") {
    Remove-Item -Path "cache\*" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "✓ Cache cleared" -ForegroundColor Green
}

# Remove all model files
if (Test-Path "models") {
    Remove-Item -Path "models\*" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "✓ Models cleared" -ForegroundColor Green
}

# Remove results
if (Test-Path "results") {
    Remove-Item -Path "results\*" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "✓ Results cleared" -ForegroundColor Green
}

# Remove logs
if (Test-Path "logs") {
    Remove-Item -Path "logs\*" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "✓ Logs cleared" -ForegroundColor Green
}

# Remove old export folder
if (Test-Path "results_export_20250918_002219") {
    Remove-Item -Path "results_export_20250918_002219" -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "✓ Old exports cleared" -ForegroundColor Green
}

# Clear model_results and saved_models (if they have content)
if (Test-Path "model_results") {
    Remove-Item -Path "model_results\*" -Force -Recurse -ErrorAction SilentlyContinue
}

if (Test-Path "saved_models") {
    Remove-Item -Path "saved_models\*" -Force -Recurse -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "All cache, models, and results cleared!" -ForegroundColor Green
Write-Host "Ready for fresh pipeline run!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Yellow
