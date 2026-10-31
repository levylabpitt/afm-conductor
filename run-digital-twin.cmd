@echo off
title AFM Conductor digital twin
rem Runs the digital twin. Installs and downloads nothing: --no-sync stops uv
rem from touching the environment, UV_PYTHON_DOWNLOADS=never stops it fetching Python.
cd /d "%~dp0"
set "PATH=%PATH%;%USERPROFILE%\.local\bin"
set UV_PYTHON_DOWNLOADS=never
where uv >nul 2>nul
if errorlevel 1 goto :notinstalled
if not exist ".venv\Scripts\python.exe" goto :notinstalled
uv run --no-sync --no-python-downloads afm-conductor %*
if errorlevel 1 (
  echo.
  echo Something went wrong. The messages above say what.
  pause
)
exit /b
:notinstalled
echo Run install-config-conductor.cmd first.
pause
exit /b 1
