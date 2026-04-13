@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"

echo Building frontend assets...
cd /d "%PROJECT_ROOT%\frontend"
call npm run build || exit /b 1

echo Starting unified AIOps Delight app on http://127.0.0.1:5001 ...
cd /d "%PROJECT_ROOT%\backend"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 app.py
  goto :eof
)

python app.py
