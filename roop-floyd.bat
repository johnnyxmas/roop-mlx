@echo off
cd /D "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Please run setup_cuda13.bat or setup_cuda12.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python run.py %*
