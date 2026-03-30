@echo off
echo ============================================================
echo  ROOP-FLOYD Setup - CUDA 13 (RTX 20-series and newer, recommended)
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

echo Installing uv...
pip install uv

echo Installing PyTorch with CUDA 13...
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo Installing core dependencies...
uv pip install numpy==1.26.4 opencv-python-headless onnx insightface albucore psutil tqdm ftfy regex pyvirtualcam

echo Installing ONNX Runtime (CUDA 13 build)...
pip install --no-cache-dir onnxruntime onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/

echo Installing Gradio and pinning pydantic...
pip install --force-reinstall pydantic==2.10.6
pip install gradio==5.13.0

echo.
echo ============================================================
echo  Setup complete! Run roop-floyd.bat to start.
echo ============================================================
pause
