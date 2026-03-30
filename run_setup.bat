@echo off
echo Starting seat zone setup...
echo.
echo Choose a detection mode:
echo   1) rectangle  - For eye-level cameras (default)
echo   2) anchor     - For CCTV / elevated cameras (RECOMMENDED)
echo   3) polygon    - For permanent CCTV installs (most accurate)
echo   4) exclusive  - Rectangle with exclusive assignment
echo.
set /p MODE_CHOICE="Enter choice (1-4, default=2 for CCTV): "

if "%MODE_CHOICE%"=="1" set MODE=rectangle
if "%MODE_CHOICE%"=="3" set MODE=polygon
if "%MODE_CHOICE%"=="4" set MODE=exclusive
if not defined MODE set MODE=anchor

echo.
echo Using mode: %MODE%
echo Draw seat zones on the camera frame, then press Enter to save.
echo.
python "%~dp0seat_occupancy.py" --setup --mode %MODE%
pause
