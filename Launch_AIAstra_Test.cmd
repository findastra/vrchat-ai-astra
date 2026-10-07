@echo off
setlocal
title AI Astra Backend Test Client
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
  echo Run Run_AIAstra.bat once, or run: cd standalone_frontend ^&^& npm install
  pause
  exit /b 1
)

where py >nul 2>&1
if not errorlevel 1 set "AI_ASTRA_BACKEND_PYTHON=py"
if not defined AI_ASTRA_BACKEND_PYTHON (
  where python >nul 2>&1
  if not errorlevel 1 set "AI_ASTRA_BACKEND_PYTHON=python"
)
if not defined AI_ASTRA_BACKEND_PYTHON (
  echo ERROR: Python 3.11 or newer was not found.
  pause
  exit /b 1
)

set "AI_ASTRA_ROOT=%ROOT%"
set "AI_ASTRA_WORKSPACE_ROOT=%ROOT%.."
echo Launching minimal AI Astra backend test client...
"%ELECTRON%" "%TEST_APP%"
set "EXIT_CODE=%errorlevel%"
endlocal & exit /b %EXIT_CODE%
