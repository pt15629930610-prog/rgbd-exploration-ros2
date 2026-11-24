@echo off
setlocal
set "ROOT=%~dp0.."
set "PYTHONUTF8=1"
chcp 65001 >nul
"E:\ANACONDA\envs\unet\python.exe" -B "%~dp0source\tools\patrol_training\check_curriculum.py" --root "%ROOT%" --experiment "%~dp0." --interactive
if errorlevel 1 pause
