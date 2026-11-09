@echo off
cd /d "%~dp0"
echo =======================================================
echo      AI Rocket Launch Simulator - Mission Control
echo =======================================================
echo.
echo Starting Streamlit Mission Control Server...
echo.
streamlit run "%~dp0app.py"
pause

