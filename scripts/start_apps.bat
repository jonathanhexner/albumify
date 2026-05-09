@echo off
REM Start Album App (backend + frontend) and Face Clustering App
REM Can be run from anywhere — auto-navigates to project root

cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run from project root or check your venv.
    pause
    exit /b 1
)

echo Starting Album App backend on port 8000...
start "Backend" cmd /k "cd /d %cd% && .venv\Scripts\python.exe -m uvicorn sim_bench.api.main:app --reload --port 8000"

timeout /t 2 /nobreak >nul

echo Starting Album App frontend on port 8501...
start "Album App" cmd /k "cd /d %cd% && .venv\Scripts\streamlit.exe run app\streamlit\main.py --server.port 8501"

echo Starting Face Clustering App on port 8502...
start "Face Clustering" cmd /k "cd /d %cd% && .venv\Scripts\streamlit.exe run app\face_clustering\main.py --server.port 8502"

echo.
echo All apps starting in separate windows:
echo   Album App:          http://localhost:8501
echo   Face Clustering:    http://localhost:8502
echo   Backend API:        http://localhost:8000/docs
