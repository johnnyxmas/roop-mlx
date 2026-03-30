@echo off
echo ============================================================
echo  ROOP-FLOYD Setup - CUDA 12.8
echo ============================================================
echo.

cd /D "%~dp0"

where python >nul 2>&1 || (echo ERROR: Python not found. Install Python 3.11 and try again. && pause && exit /b 1)
where ffmpeg >nul 2>&1 || echo WARNING: ffmpeg not found on PATH. Video processing will not work.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 ( echo ERROR: Failed to create venv. && pause && exit /b 1 )

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install uv
pip install --force-reinstall pydantic==2.10.6
uv pip install -r requirements.txt
pip install gradio==5.13.0

echo.
echo ============================================================
echo  Setup complete! Run roop-floyd.bat to start.
echo ============================================================
pause
