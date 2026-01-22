@echo off
echo Starting documentation server...
echo.
echo Open your browser to: http://localhost:8000/VIEW_DOCS.html
echo.
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
python -m http.server 8000
