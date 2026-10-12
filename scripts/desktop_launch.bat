@echo off
setlocal
cd /d "%~dp0\.."

echo =====================================================================
echo  Trading AI Assistant - Private Desktop Launch
echo =====================================================================

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" scripts\desktop_launch.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Launch failed. Please review the output above.
    pause
)
