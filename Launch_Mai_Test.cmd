@echo off
setlocal
title Mai Backend Test Client
set "ROOT=%~dp0"
set "TEST_APP=%ROOT%testing_frontend"
set "ELECTRON=%ROOT%standalone_frontend\node_modules\electron\dist\electron.exe"

if not exist "%TEST_APP%\main.js" (
  echo ERROR: testing_frontend is missing.
  pause
  exit /b 1
)
if not exist "%ELECTRON%" (
  echo ERROR: Electron is not installed yet.
  echo Run Run_Mai.bat once, or run: cd standalone_frontend ^&^& npm install
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
set "MAI_WORKSPACE_ROOT=%ROOT%.."
echo Launching minimal Mai backend test client...
"%ELECTRON%" "%TEST_APP%"
set "EXIT_CODE=%errorlevel%"
endlocal & exit /b %EXIT_CODE%
