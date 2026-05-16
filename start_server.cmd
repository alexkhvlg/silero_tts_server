@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python server.py --host 0.0.0.0 --port 5000 --device cpu
