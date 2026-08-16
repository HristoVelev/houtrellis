@echo off
rem =====================================================================
rem HouTrellis: Standalone Multiplatform Windows Launcher
rem =====================================================================
set DIR=%~dp0

rem Execute the modular CLI module using our virtual environment's Python interpreter
"%DIR%backend\venv\Scripts\python.exe" "%DIR%backend\cli.py" %*
