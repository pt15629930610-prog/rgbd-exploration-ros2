@echo off
setlocal
cd /d "%~dp0"
if not exist "Qwen3VL-2B-Instruct-Q4_K_M.gguf" goto missing
if not exist "mmproj-Qwen3VL-2B-Instruct-F16.gguf" goto missing
if not exist "door_log_sync.py" goto missing
set "LLAMA_CACHE=%~dp0cache"
set "QWEN_PY=E:\ANACONDA\envs\unet\python.exe"
if not exist "%QWEN_PY%" set "QWEN_PY=python"
echo Greenhouse online exploration - local Qwen, Vulkan batch 128 / microbatch 32, flash attention off
echo Model: http://192.168.19.1:18080   Logs: %~dp0door_nav_logs
"%QWEN_PY%" -B "%~dp0door_log_sync.py" receive --bind 192.168.19.1 --run "gpu-bin\llama-server.exe" -m "Qwen3VL-2B-Instruct-Q4_K_M.gguf" --mmproj "mmproj-Qwen3VL-2B-Instruct-F16.gguf" --device Vulkan0 -ngl 99 -fa off -b 128 -ub 32 -c 4096 -np 1 --alias Qwen3-VL-2B-Instruct --host 192.168.19.1 --port 18080
if errorlevel 1 pause
exit /b
:missing
echo Place this launcher in the existing Qwen3-VL-2B model and log receiver directory.
pause
exit /b 1
