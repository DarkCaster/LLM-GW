@echo off
setlocal

set "script_dir=%~dp0"
set "venv_dir=%script_dir%venv"

if not exist "%venv_dir%" (
    python -m venv "%venv_dir%"
)

REM more information for installing vllm on windows:
REM https://github.com/SystemPanic/vllm-windows
REM https://github.com/woct0rdho/triton-windows/releases

"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade pip
"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade --extra-index-url="https://download.pytorch.org/whl/cu124" -r "%script_dir%win-requirements.txt"

endlocal
