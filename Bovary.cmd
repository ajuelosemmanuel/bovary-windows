@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Set up Bovary in .venv using the README quick start first.
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m bovary.gui
