@echo off
title AFM Conductor hardware install
rem One-time setup of an Asylum instrument PC. Asks before each step.
rem Run install-config-conductor.cmd first.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-config-hardware.ps1" %*
echo.
pause
