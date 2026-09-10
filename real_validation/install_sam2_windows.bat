@echo off
cd /d "%~dp0.."
python -m pip install -r real_validation/requirements-sam2.txt
if errorlevel 1 (
  echo SAM2 dependency installation failed. Check the error above.
) else (
  echo SAM2 dependencies installed. Run real_validation\run_gui.bat.
)
pause
