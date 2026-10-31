@echo off
title AFM Conductor hardware twin
rem Every-day start on the instrument PC: the service in its own window, then
rem the Igor bridge. Installs and downloads nothing (--no-sync, no Python downloads).
cd /d "%~dp0"
set "PATH=%PATH%;%USERPROFILE%\.local\bin"
set UV_PYTHON_DOWNLOADS=never
where uv >nul 2>nul
if errorlevel 1 goto :noconductor
if not exist ".venv\Scripts\python.exe" goto :noconductor
if not exist ".venv\Lib\site-packages\win32com" goto :nohardware
set AFM_LITHO_BACKEND=hardware
set AFM_LITHO_WIRE_VERSION=3
start "AFM Conductor service" cmd /k uv run --no-sync --no-python-downloads --extra hardware afm-conductor
echo.
echo The service is starting in its own window.
echo Open Igor with the Asylum software loaded, then press Enter.
pause >nul
uv run --no-sync --no-python-downloads --extra hardware scripts\start_bridge.py
if errorlevel 1 (
  echo.
  echo The bridge did not start. The messages above say why.
)
echo.
pause
exit /b
:noconductor
echo Run install-config-conductor.cmd first, then install-config-hardware.cmd.
pause
exit /b 1
:nohardware
echo Run install-config-hardware.cmd first.
pause
exit /b 1
