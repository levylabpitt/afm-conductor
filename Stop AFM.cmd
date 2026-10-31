@echo off
title Stop AFM
rem Double-click to Stop: parks the tip. Sends the Stop to the running keeper
rem (port 9211) and shows the reply; if the keeper does not answer it sends to the
rem host stop port (9210). Uses uv without installing anything; without uv it uses .venv from an earlier setup.
cd /d "%~dp0"
set "PATH=%PATH%;%USERPROFILE%\.local\bin"
where uv >nul 2>nul
if not errorlevel 1 (
  uv run --no-sync --no-python-downloads -m afm_litho.keeper stop
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m afm_litho.keeper stop
) else (
  echo uv and .venv are both missing - run Start Twin.cmd first
)
echo.
pause
