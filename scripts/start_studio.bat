@echo off
REM ============================================================
REM  Image Analysis Studio (spec-094)
REM  Point at a photo folder, run quality + geo methods, browse
REM  the results. Opens at http://localhost:8503
REM ============================================================
cd /d "%~dp0\.."
if not exist ".venv\Scripts\streamlit.exe" (
    echo ERROR: .venv not found. Run this from the project root's venv.
    pause
    exit /b 1
)
echo.
echo Starting Image Analysis Studio at http://localhost:8503
echo (a browser tab should open; Ctrl+C in this window to stop)
echo.
.venv\Scripts\streamlit.exe run app\image_studio\main.py --server.port 8503
pause
