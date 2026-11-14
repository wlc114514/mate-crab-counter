@echo off
rem Install dependencies for the crab counter (Python, PyTorch, rfdetr) and create a desktop shortcut.
rem Options: -Cpu (force CPU PyTorch)  -NoShortcut
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
