# Simple cleanup script that's guaranteed to work

# Create backup
$backupFile = "bci_project_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
Write-Host "📦 Creating backup: $backupFile" -ForegroundColor Cyan
Compress-Archive -Path .\* -DestinationPath $backupFile -Force

# Create a folder for files to review
$trashFolder = "to_review_$(Get-Date -Format 'yyyyMMdd')"
New-Item -ItemType Directory -Force -Path $trashFolder | Out-Null

# Files to keep (everything else will be moved to review folder)
$keepFiles = @(
    "app_enhanced.py",
    "enhanced_styles.py",
    "app.py",
    "run_pipeline.py",
    "config.yaml",
    "music_map.yaml",
    "requirements.txt",
    "README.md",
    "PROJECT_REPORT.md",
    "modules"
)

# Move all Python files not in keep list
Get-ChildItem -Path . -Filter "*.py" -File | Where-Object { 
    $keepFiles -notcontains $_.Name 
} | ForEach-Object {
    $dest = Join-Path $trashFolder $_.Name
    Move-Item $_.FullName $dest -Force
    Write-Host "Moved: $($_.Name)" -ForegroundColor Yellow
}

# Move test and diagnostic files
@("test_*", "verify_*", "diagnose_*") | ForEach-Object {
    Get-ChildItem -Path . -Filter $_ -File | ForEach-Object {
        $dest = Join-Path $trashFolder $_.Name
        Move-Item $_.FullName $dest -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "`n✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "   - Backup created: $backupFile" -ForegroundColor Cyan
Write-Host "   - Files moved to review: $trashFolder" -ForegroundColor Cyan
Write-Host "`n🔍 Review the files in '$trashFolder' before deleting them permanently." -ForegroundColor Yellow
