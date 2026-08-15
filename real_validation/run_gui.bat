@echo off
cd /d "%~dp0.."
python -m real_validation.main
if errorlevel 1 pause
