@echo off
setlocal
title AI Astra Voice
cd /d "%~dp0"

if exist "%~dp0..\.venv\Scripts\python.exe" (set PY="%~dp0..\.venv\Scripts\python.exe"& goto run)
where py >nul 2>&1
if not errorlevel 1 (set "PY=py -3") else (set "PY=python")

:run
if "%ANTHROPIC_API_KEY%"=="" (
  echo ANTHROPIC_API_KEY is not set, so AI Astra will run in offline mode
  echo ^(you'll see her goal and strategy, but no replies^).
  echo See README.md, step 1, to add your key.
  echo.
  %PY% ai_astra_voice.py --offline
) else (
  %PY% ai_astra_voice.py %*
)
pause
