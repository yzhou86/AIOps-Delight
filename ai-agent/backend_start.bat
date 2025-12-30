@echo off
echo 启动AI Agent后端服务...
cd /d "%~dp0backend"
python app.py
pause