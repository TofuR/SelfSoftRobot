@echo off
cd /d "%~dp0"
python main_validation.py
if errorlevel 1 pause
