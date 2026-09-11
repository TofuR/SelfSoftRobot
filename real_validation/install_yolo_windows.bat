@echo off
cd /d "%~dp0.."
python -m pip install -r real_validation/requirements-yolo.txt
if errorlevel 1 pause
