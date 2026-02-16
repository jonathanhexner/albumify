@echo off
cd /d D:\sim-bench
call .venv\Scripts\activate.bat
python scripts/benchmark_face_clustering.py --album-path "D:\Budapest2025_Google"
pause
