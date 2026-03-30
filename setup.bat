@echo off
echo ============================================
echo   Seat Occupancy Demo - One-Time Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/2] Installing required packages...
pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install packages.
    pause
    exit /b 1
)

echo.
echo [2/2] Downloading YOLOv8-nano model (first time only, ~6 MB)...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('[OK] Model ready.')"

echo.
echo ============================================
echo   Setup complete! You can now run:
echo     run_setup.bat    - Draw seat zones
echo     run_detect.bat   - Start detection
echo ============================================
pause
