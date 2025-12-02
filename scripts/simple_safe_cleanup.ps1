# Simple Safe Cleanup Script

# 1. Create backup
$backupFile = "bci_project_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
Write-Host "Creating backup: $backupFile" -ForegroundColor Cyan
Compress-Archive -Path .\* -DestinationPath $backupFile -Force

# 2. Create review folder
$reviewFolder = "to_review_$(Get-Date -Format 'yyyyMMdd')
New-Item -ItemType Directory -Path $reviewFolder -Force | Out-Null

# 3. Define patterns to move (not delete)
$patterns = @("test_*.py", "verify_*.py", "diagnose_*.py", "*.bak", "*.tmp", "*.log", "*.old", "*.backup")

# 4. Move files to review folder
foreach ($pattern in $patterns) {
    Get-ChildItem -Path . -Filter $pattern -File -Recurse | ForEach-Object {
        $dest = Join-Path $reviewFolder $_.Name
        $destDir = [System.IO.Path]::GetDirectoryName($dest)
        if (-not (Test-Path $destDir)) { 
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null 
        }
        Move-Item $_.FullName $dest -Force -ErrorAction SilentlyContinue
        Write-Host "Moved: $($_.Name)" -ForegroundColor Yellow
    }
}

# 5. Show summary
$movedCount = (Get-ChildItem -Path $reviewFolder -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host "`nCleanup Complete!" -ForegroundColor Green
Write-Host "Backup created: $backupFile" -ForegroundColor Cyan
Write-Host "Files moved to review: $reviewFolder ($movedCount items)" -ForegroundColor Cyan

# 6. Show essential folders
Write-Host "`nProtected folders (not touched):" -ForegroundColor Blue
@("modules", "music", "information", "models", "data", "results") | ForEach-Object {
    if (Test-Path $_) {
        Write-Host "- $_" -ForegroundColor Blue 
    }
}

Write-Host "`nReview the files in '$reviewFolder' before deleting them." -ForegroundColor Yellow
