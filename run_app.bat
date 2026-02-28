@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Create venv first: python -m venv venv
    pause
    exit /b 1
)
echo Starting Streamlit. Keep this window open.
echo Open in browser: http://localhost:8501
echo.
python -m streamlit run app.py --server.headless true
echo.
echo Server exited with code: %errorlevel%
pause
