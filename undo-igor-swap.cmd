@echo off
title AFM Conductor: undo the Igor 6.37 swap
rem Puts the Asylum Igor.exe (6.38B01), kept as Igor.exe.6.38-backup, back in place.
rem Igor must be closed; Windows asks for administrator permission when the folder needs it.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-config-hardware.ps1" -UndoIgor %*
echo.
pause
