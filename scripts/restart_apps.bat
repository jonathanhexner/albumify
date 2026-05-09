@echo off
REM Restart Album App (backend + frontend) and Face Clustering App
REM Closes old cmd windows by title, kills ports, launches fresh

cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run from project root or check your venv.
    pause
    exit /b 1
)

echo Closing old app windows...
taskkill /FI "WINDOWTITLE eq Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Album App*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Face Clustering*" /F >nul 2>&1

echo Stopping processes on ports 8000, 8501, 8502...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8502 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul

echo Starting Album App backend on port 8000...
start "Backend" cmd /k "cd /d %cd% && .venv\Scripts\python.exe -m uvicorn sim_bench.api.main:app --reload --port 8000"

timeout /t 2 /nobreak >nul

echo Starting Album App frontend on port 8501...
start "Album App" cmd /k "cd /d %cd% && .venv\Scripts\streamlit.exe run app\streamlit\main.py --server.port 8501"

echo Starting Face Clustering App on port 8502...
start "Face Clustering" cmd /k "cd /d %cd% && .venv\Scripts\streamlit.exe run app\face_clustering\main.py --server.port 8502"

echo.
echo All apps restarted:
echo   Album App:          http://localhost:8501
echo   Face Clustering:    http://localhost:8502
echo   Backend API:        http://localhost:8000/docs
