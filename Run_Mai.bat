@echo off
setlocal
title Mai Standalone

rem Safe launcher: this file does not run PowerShell, install packages, or start Newtype.
set "ROOT=%~dp0"
set "ELECTRON=%ROOT%standalone_frontend\node_modules\electron\dist\electron.exe"

if not exist "%ROOT%standalone_frontend\main.js" (
  echo ERROR: standalone_frontend\main.js is missing.
  pause
  exit /b 1
)

if not exist "%ELECTRON%" (
  echo ERROR: Electron is not installed.
  echo Run this manually from the standalone_frontend folder:
  echo   npm install
  pause
  exit /b 1
)

where py >nul 2>&1
if not errorlevel 1 set "MAI_BACKEND_PYTHON=py"
if not defined MAI_BACKEND_PYTHON (
  where python >nul 2>&1
  if not errorlevel 1 set "MAI_BACKEND_PYTHON=python"
)
if not defined MAI_BACKEND_PYTHON (
  echo ERROR: Python 3.11 or newer was not found.
  pause
  exit /b 1
)

set "MAI_ROOT=%ROOT%"
for %%I in ("%ROOT%..") do set "MAI_WORKSPACE_ROOT=%%~fI"

echo Starting Mai Standalone without legacy launchers...
"%ELECTRON%" "%ROOT%standalone_frontend"
set "EXIT_CODE=%errorlevel%"
endlocal & exit /b %EXIT_CODE%
