# Force clear ALL caches and restart
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "FORCE CACHE CLEAR & RESTART" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

Set-Location "c:\Arman\Projects\BCI PROJECT"

Write-Host "`n[1/5] Stopping Streamlit..." -ForegroundColor Cyan
Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "      Done" -ForegroundColor Green

Write-Host "[2/5] Deleting Streamlit cache..." -ForegroundColor Cyan
Remove-Item -Path .streamlit -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[3/5] Deleting Python cache..." -ForegroundColor Cyan
Get-ChildItem -Path . -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "*.pyc" -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[4/5] Deleting session state..." -ForegroundColor Cyan
Remove-Item -Path app_state.db -Force -ErrorAction SilentlyContinue
Remove-Item -Path .streamlit/cache -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "      Done" -ForegroundColor Green

Write-Host "[5/5] Starting fresh app on port 8506..." -ForegroundColor Yellow
Start-Process python -ArgumentList "-m","streamlit","run","app.py","--server.port","8506" -WindowStyle Hidden

Start-Sleep -Seconds 5

Write-Host "`n=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "DONE! App starting on port 8506" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "`nOpen: http://localhost:8506" -ForegroundColor White
