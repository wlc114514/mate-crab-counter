@echo off
rem Install dependencies for the crab counter (Python, PyTorch, rfdetr) and register the vision service for QGroundControl Vision.
rem Options: -Cpu (force CPU PyTorch)  -Shortcut (also create the old "QGC + crab button" desktop shortcut)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
