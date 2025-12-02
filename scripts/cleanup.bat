@echo off
echo Creating backup...
set timestamp=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%
set backupFile=bci_backup_%timestamp%.zip
powershell -Command "Compress-Archive -Path .\* -DestinationPath %backupFile% -Force"
echo Backup created: %backupFile%

set reviewFolder=to_review_%date:~-4%%date:~3,2%%date:~0,2%
if not exist "%reviewFolder%" mkdir "%reviewFolder%"

echo Moving test files to %reviewFolder%...
for %%p in (test_*.py verify_*.py diagnose_*.py *.bak *.tmp *.log *.old *.backup) do (
    if exist "%%p" (
        echo Moving: %%p
        move /Y "%%p" "%reviewFolder%\" >nul
    )
)

echo.
echo Cleanup complete!
echo - Backup: %backupFile%
echo - Review folder: %reviewFolder%
echo.
echo Protected folders (not modified):
echo - modules
echo - music
echo - information
echo - models
echo - data
echo - results
echo.
pause
