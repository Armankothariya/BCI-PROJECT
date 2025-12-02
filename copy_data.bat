@echo off
xcopy "datasets\kaggle_temp\emotion.csv" "data\" /Y
xcopy "results\*.csv" "data\" /Y
echo Files copied successfully!
pause
