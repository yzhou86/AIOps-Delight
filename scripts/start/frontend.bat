@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"

echo Building frontend assets...
cd /d "%PROJECT_ROOT%\frontend"
call npm run build
