@echo off
setlocal
cd /d "%~dp0"
title Qualcomm S24 ASR Benchmark
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo Benchmark did not finish. Fix the error if needed, then run this BAT again; checkpoints will resume completed work.
) else (
  echo Finished successfully.
)
pause
exit /b %ERR%
