@echo off
echo Removing review folder...
if exist "to_review_2025 1Th" (
    rmdir /s /q "to_review_2025 1Th"
    echo Review folder has been removed.
) else (
    echo Review folder not found or already removed.
)
pause
