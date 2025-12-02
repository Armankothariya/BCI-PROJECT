# safe_cleanup_keep_frontend.ps1
# Run this script in your project directory

# Files to NEVER delete
$protectedFiles = @(
    "app_enhanced.py",
    "enhanced_styles.py",
    "app.py",
    "run_pipeline.py",
    "config.yaml",
    "music_map.yaml",
    "requirements.txt",
    "README.md",
    "PROJECT_REPORT.md",
    "modules/*"
)

# Create backup
$backupFile = "bci_project_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
Write-Host "📦 Creating backup: $backupFile" -ForegroundColor Cyan
Compress-Archive -Path .\* -DestinationPath $backupFile -Force

# Create a folder for files to delete
$trashFolder = "to_review_$(Get-Date -Format 'yyyyMMdd')"
New-Item -ItemType Directory -Force -Path $trashFolder | Out-Null
$logFile = "$trashFolder/cleanup_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

# Start logging
"=== BCI Project Cleanup Log $(Get-Date) ===" | Out-File -FilePath $logFile
"Backup created: $backupFile" | Out-File -FilePath $logFile -Append

function Safe-Move {
    param($pattern, $description)
    $files = Get-ChildItem -Path . -Filter $pattern -File -Recurse -ErrorAction SilentlyContinue | 
             Where-Object { $protectedFiles -notcontains $_.Name -and 
                           $protectedFiles -notcontains "modules/$($_.Name)" }
    
    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring((Get-Location).Path.Length + 1)
        $dest = Join-Path $trashFolder $relativePath
        $destDir = [System.IO.Path]::GetDirectoryName($dest)
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
        Move-Item $file.FullName $dest -Force
        "Moved: $relativePath" | Out-File -FilePath $logFile -Append
        Write-Host "  → Moved: $relativePath" -ForegroundColor Yellow
    }
    if ($files) {
        Write-Host "✓ $($files.Count) $description moved to $trashFolder" -ForegroundColor Green
    }
}

# 1. Move test and verification scripts (except protected files)
Write-Host "`n🔍 Moving test and verification scripts..." -ForegroundColor Cyan
Safe-Move "test_*.py" "test scripts"
Safe-Move "verify_*.py" "verification scripts"
Safe-Move "diagnose_*.py" "diagnostic scripts"

# 2. Move old versions of frontend files (keep only the latest)
$frontendFiles = @("app.py", "app_enhanced.py", "enhanced_styles.py")
foreach ($file in $frontendFiles) {
    $backups = Get-ChildItem -Path . -Filter "$($file).bak*" -File
    foreach ($backup in $backups) {
        $dest = Join-Path $trashFolder $backup.Name
        Move-Item $backup.FullName $dest -Force
        "Moved backup: $($backup.Name)" | Out-File -FilePath $logFile -Append
        Write-Host "  → Moved backup: $($backup.Name)" -ForegroundColor Yellow
    }
}

# 3. Clean cache and temporary files (preserving frontend cache)
Write-Host "`n🗑️  Cleaning cache and temporary files..." -ForegroundColor Cyan
if (Test-Path "__pycache__") {
    # Keep only frontend-related cache
    $cacheFiles = Get-ChildItem -Path "__pycache__" -File | 
                 Where-Object { $_.Name -notmatch "app.*\.pyc|enhanced.*\.pyc" }
    foreach ($file in $cacheFiles) {
        $dest = Join-Path $trashFolder "__pycache__\$($file.Name)"
        if (-not (Test-Path (Join-Path $trashFolder "__pycache__"))) {
            New-Item -ItemType Directory -Path (Join-Path $trashFolder "__pycache__") | Out-Null
        }
        Move-Item $file.FullName $dest -Force
        "Moved cache: __pycache__\$($file.Name)" | Out-File -FilePath $logFile -Append
    }
}

# 4. Clean results folder (keep structure, move only logs/tmp)
if (Test-Path "results") {
    $resultFiles = Get-ChildItem -Path "results" -File -Recurse | 
                  Where-Object { $_.Extension -match '\.log|\.tmp' }
    foreach ($file in $resultFiles) {
        $relativePath = $file.FullName.Substring((Get-Location).Path.Length + 1)
        $dest = Join-Path $trashFolder $relativePath
        $destDir = [System.IO.Path]::GetDirectoryName($dest)
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
        Move-Item $file.FullName $dest -Force
        "Moved result: $relativePath" | Out-File -FilePath $logFile -Append
    }
    if ($resultFiles) {
        Write-Host "✓ Moved $($resultFiles.Count) temporary result files" -ForegroundColor Green
    }
}

# Show summary
$movedCount = (Get-ChildItem -Path $trashFolder -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host "`n✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "   - Backup created: $backupFile" -ForegroundColor Cyan
Write-Host "   - Files moved to review: $trashFolder ($movedCount items)" -ForegroundColor Cyan
Write-Host "   - Log file: $logFile" -ForegroundColor Cyan

# Show protected files that were kept
Write-Host "`n🛡️  Protected files (not touched):" -ForegroundColor Blue
$protectedFiles | ForEach-Object { 
    if ($_ -ne "modules/*") {
        Write-Host "   - $_" -ForegroundColor Blue 
    } else {
        Write-Host "   - modules/ (all files)" -ForegroundColor Blue
    }
}

Write-Host "`n🔍 Review the files in '$trashFolder' before deleting them permanently." -ForegroundColor Yellow
Write-Host "   To undo: Copy files back from '$trashFolder' to their original locations" -ForegroundColor Yellow
