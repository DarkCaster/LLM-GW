@echo off
setlocal

set "script_dir=%~dp0"
set "venv_dir=%script_dir%venv"

if not exist "%venv_dir%" (
    python -m venv "%venv_dir%"
)

REM You may need to insyall microsoft c++ build tools (from visual studio, may be installed separately), windows 10/11 SDK, CUDA SDK

"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade pip
"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url="https://download.pytorch.org/whl/cu124"
"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade "https://github.com/SystemPanic/vllm-windows/releases/download/v0.8.5/vllm-0.8.5+cu124-cp312-cp312-win_amd64.whl"

endlocal
