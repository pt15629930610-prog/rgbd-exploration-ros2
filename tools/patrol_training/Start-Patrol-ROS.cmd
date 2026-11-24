@echo off
setlocal
cd /d "%~dp0..\.."
if not exist "patrol_training\route-v3\adapter\patrol-lora.gguf" (
  echo Missing route-v3 adapter.
  pause
  exit /b 1
)
"E:\ANACONDA\envs\unet\python.exe" -B door_log_sync.py receive --bind 192.168.19.1 --run gpu-bin\llama-server.exe -m Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen3VL-2B-Instruct-F16.gguf --device Vulkan0 -ngl 99 -fa off -b 128 -ub 32 -c 4096 -np 1 --alias Qwen3-VL-2B-Instruct --lora-scaled "patrol_training\route-v3\adapter\patrol-lora.gguf:0" --lora-init-without-apply --host 192.168.19.1 --port 18080
if errorlevel 1 pause
