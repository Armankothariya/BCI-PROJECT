# cleanup_temp_files.ps1

# 1. Folders to completely remove (recreatable)
$foldersToDelete = @(
    ".ipynb_checkpoints",
    ".vscode",
    "cache",
    "__pycache__",
    "*.egg-info"
)

# 2. Folders to clean (keep but remove temp files)
$foldersToClean = @(
    "models",
    "results",
    "logs",
    "utils"
)

# 3. File patterns to remove from kept folders
$tempFilePatterns = @(
    "*.tmp",
    "*.temp",
    "*.log",
    "*.bak",
    "*.backup",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "Thumbs.db",
    "desktop.ini"
)

Write-Host "🧹 Starting cleanup..." -ForegroundColor Cyan

# 4. Delete entire folders that can be recreated
Write-Host "`n🗑️  Removing temporary folders..." -ForegroundColor Yellow
foreach ($folder in $foldersToDelete) {
    Get-ChildItem -Path . -Directory -Filter $folder -Recurse | ForEach-Object {
        Write-Host "   Removing: $($_.FullName)" -ForegroundColor Red
        Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 5. Clean temp files from kept folders
Write-Host "`n🧽 Cleaning temporary files from kept folders..." -ForegroundColor Yellow
foreach ($folder in $foldersToClean) {
    if (Test-Path $folder) {
        Write-Host "   Cleaning: $folder" -ForegroundColor Cyan
        foreach ($pattern in $tempFilePatterns) {
            Get-ChildItem -Path $folder -Include $pattern -File -Recurse | ForEach-Object {
                Write-Host "      Removing: $($_.FullName)" -ForegroundColor DarkGray
                Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
            }
        }
        # Remove empty directories
        Get-ChildItem -Path $folder -Directory -Recurse | 
            Where-Object { (Get-ChildItem -Path $_.FullName -Recurse -File).Count -eq 0 } |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 6. Clean root directory temp files
Write-Host "`n🧼 Cleaning root directory temporary files..." -ForegroundColor Yellow
foreach ($pattern in $tempFilePatterns) {
    Get-ChildItem -Path . -Filter $pattern -File | ForEach-Object {
        Write-Host "   Removing: $($_.Name)" -ForegroundColor DarkGray
        Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "`n✨ Cleanup complete! Your project is now tidy." -ForegroundColor Green
