@echo off
rem Start the Docker stack, auto-detecting the GPU override.
rem Usage: run_docker.bat [up|down|build|status]
rem The real logic lives in docker\up.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0docker\up.ps1" %*