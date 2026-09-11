@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto dependencies
py -3.10 -c "import struct; assert struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto python_fallback
py -3.10 -m venv .venv
if errorlevel 1 goto failed
goto dependencies
:python_fallback
python -c "import sys,struct; assert sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto missing_python
python -m venv .venv
if errorlevel 1 goto failed
:dependencies
".venv\Scripts\python.exe" -m pip install -r real_validation\requirements.txt
if errorlevel 1 goto failed
echo Installation complete. Start the App with run.bat.
pause
exit /b 0
:missing_python
echo Please install Python 3.10 64-bit, then run install.bat again.
pause
exit /b 1
:failed
echo Installation failed. Check the error above and your network connection.
pause
exit /b 1
