@echo off
setlocal
cd /d "%~dp0\.."

echo =====================================================================
echo  Trading AI Assistant - Safe Application Update Runner
echo =====================================================================

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" scripts\safe_update.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Update pre-flight checks failed. Rollback state preserved.
    pause
)
