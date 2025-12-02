# Safe cleanup script that preserves all essential files and folders

# 1. Create backup
$backupFile = "bci_project_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
Write-Host "📦 Creating backup: $backupFile" -ForegroundColor Cyan
Compress-Archive -Path .\* -DestinationPath $backupFile -Force

# 2. Create review folder
$reviewFolder = "to_review_$(Get-Date -Format 'yyyyMMdd')
New-Item -ItemType Directory -Path $reviewFolder -Force | Out-Null

# 3. Define what to keep
$keepFiles = @(
    # Core application
    "app_enhanced.py",
    "enhanced_styles.py",
    "app.py",
    "run_pipeline.py",
    "config.yaml",
    "music_map.yaml",
    "requirements.txt",
    "README.md",
    "PROJECT_REPORT.md",
    "simple_cleanup.ps1",
    "safe_cleanup_keep_frontend.ps1",
    "get_latest_logs.py"
)

# 4. Folders to keep (all files inside these are safe)
$keepFolders = @(
    "modules",
    "music",
    "information",
    "models",
    "data",
    "results",
    "cache",
    "static",
    "templates",
    "docs"
)

# 5. File patterns to move to review
$patternsToMove = @(
    "test_*.py",
    "verify_*.py",
    "diagnose_*.py",
    "*.bak",
    "*.tmp",
    "*.log",
    "*.old",
    "*.backup"
)

# 6. Move files matching patterns (if not in keep files/folders)
foreach ($pattern in $patternsToMove) {
    Get-ChildItem -Path . -Filter $pattern -File -Recurse | ForEach-Object {
        $relativePath = $_.FullName.Substring((Get-Location).Path.Length + 1)
        $inKeepFolder = $false
        
        # Check if file is in a folder we want to keep
        foreach ($folder in $keepFolders) {
            if ($relativePath.StartsWith($folder + "\")) {
                $inKeepFolder = $true
                break
            }
        }
        
        if (-not $inKeepFolder -and ($keepFiles -notcontains $_.Name)) {
            $dest = Join-Path $reviewFolder $relativePath
            $destDir = [System.IO.Path]::GetDirectoryName($dest)
            if (-not (Test-Path $destDir)) { 
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null 
            }
            Move-Item $_.FullName $dest -Force
            Write-Host "  → Moved: $relativePath" -ForegroundColor Yellow
        }
    }
}

# 7. Show summary
$movedCount = (Get-ChildItem -Path $reviewFolder -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host "`n✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "   - Backup created: $backupFile" -ForegroundColor Cyan
Write-Host "   - Files moved to review: $reviewFolder ($movedCount items)" -ForegroundColor Cyan

# 8. Show protected folders
Write-Host "`n🛡️  Protected folders (not touched):" -ForegroundColor Blue
$keepFolders | Sort-Object | ForEach-Object { 
    if (Test-Path $_) {
        Write-Host "   - $_" -ForegroundColor Blue 
    }
}

Write-Host "`n🔍 Review the files in '$reviewFolder' before deleting them." -ForegroundColor Yellow
Write-Host "   To undo: Copy files back from '$reviewFolder' to their original locations" -ForegroundColor Yellow

# 9. Create a verification script
$verifyScript = @'
# verify_cleanup.ps1
Write-Host "🔍 Verifying essential files and folders..." -ForegroundColor Cyan

$essentialFiles = @(
    "app_enhanced.py",
    "enhanced_styles.py",
    "app.py",
    "run_pipeline.py",
    "config.yaml",
    "music_map.yaml"
)

$essentialFolders = @(
    "modules",
    "music",
    "information",
    "models"
)

Write-Host "`n✅ Essential files found:" -ForegroundColor Green
$allOk = $true
foreach ($file in $essentialFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file (MISSING)" -ForegroundColor Red
        $allOk = $false
    }
}

Write-Host "`n📁 Essential folders found:" -ForegroundColor Green
foreach ($folder in $essentialFolders) {
    if (Test-Path $folder) {
        $count = (Get-ChildItem -Path $folder -Recurse -File).Count
        Write-Host ("   ✓ {0,-15} ({1} files)" -f $folder, $count) -ForegroundColor Green
    } else {
        Write-Host "   ✗ $folder (MISSING)" -ForegroundColor Red
        $allOk = $false
    }
}

if ($allOk) {
    Write-Host "`n✅ All essential files and folders are present!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Some files or folders are missing. Check the list above." -ForegroundColor Red
}
'@

# Save verification script
$verifyScript | Out-File -FilePath "verify_cleanup.ps1" -Encoding utf8
Write-Host "`n🔍 Created verification script: verify_cleanup.ps1" -ForegroundColor Cyan
Write-Host "   Run this script to verify all essential files are still present." -ForegroundColor Cyan
