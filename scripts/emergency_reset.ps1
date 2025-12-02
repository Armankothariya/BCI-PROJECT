# Emergency Reset - Clear all cache and restart fresh
Write-Host "🚨 EMERGENCY RESET - Clearing all caches" -ForegroundColor Red

# Stop any running Streamlit processes
Write-Host "Stopping Streamlit processes..." -ForegroundColor Yellow
Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process -Force

# Clear cache directory
Write-Host "Clearing cache directory..." -ForegroundColor Yellow
Remove-Item -Path "cache\*" -Force -Recurse -ErrorAction SilentlyContinue

# Clear Streamlit cache
Write-Host "Clearing Streamlit cache..." -ForegroundColor Yellow
if (Test-Path "$env:USERPROFILE\.streamlit\cache") {
    Remove-Item -Path "$env:USERPROFILE\.streamlit\cache\*" -Force -Recurse -ErrorAction SilentlyContinue
}

Write-Host "" -ForegroundColor Green
Write-Host "✅ RESET COMPLETE!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: python run_pipeline.py" -ForegroundColor White
Write-Host "2. Wait for training to complete" -ForegroundColor White
Write-Host "3. Run: streamlit run app.py" -ForegroundColor White
Write-Host "4. In app: Click 'Run System Preflight'" -ForegroundColor White
Write-Host "5. Start fresh demo" -ForegroundColor White
