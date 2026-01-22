@echo off
echo ================================================================================
echo                  RESTARTING ALBUM APP WITH YOUR MODELS
echo ================================================================================
echo.
echo Your trained models are now active:
echo   - AVA Aesthetic Model (50%% selection weight)
echo   - Siamese Comparison Model (tiebreaking)
echo.
echo Look for these in the startup logs:
echo   "INFO - Loaded AVA model"
echo   "INFO - Loaded Siamese comparison model"
echo.
echo ================================================================================
echo.
cd /d "%~dp0"
streamlit run app/album/main.py
