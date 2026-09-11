@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run install.bat first.
  pause
  exit /b 1
)
set "REAL_VALIDATION_RESULTS=%~dp0results"
set "YOLO_CONFIG_DIR=%~dp0.venv\cache\ultralytics"
set "MPLCONFIGDIR=%~dp0.venv\cache\matplotlib"
set "PYTHONDONTWRITEBYTECODE=1"
".venv\Scripts\python.exe" -m real_validation.main
if errorlevel 1 (
  pause
  exit /b 1
)
