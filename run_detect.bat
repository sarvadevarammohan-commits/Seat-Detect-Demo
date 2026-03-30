@echo off
echo Starting seat occupancy detection...
echo Press Q or Esc to quit. Press S to save layout image.
echo.
python "%~dp0seat_occupancy.py" --show-fps
pause
