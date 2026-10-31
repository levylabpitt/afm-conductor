@echo off
title AFM Conductor software install
rem Installs the software (uv, Python 3.12, packages). Asks first; starts nothing.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-config-conductor.ps1"
echo.
pause
