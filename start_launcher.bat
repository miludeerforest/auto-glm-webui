@echo off
chcp 65001 > nul
echo Starting Open-AutoGLM Web Launcher...

:: Set encoding variables
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Activate Conda environment
call conda activate autoglm
if %errorlevel% neq 0 (
    echo Error: Failed to activate conda environment 'autoglm'.
    echo Please make sure you have created the environment.
    pause
    exit /b 1
)

:: Open browser
start http://127.0.0.1:8000

:: Start the server
python launcher_app.py

pause
