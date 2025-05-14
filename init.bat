@echo off
setlocal

set "script_dir=%~dp0"
set "venv_dir=%script_dir%venv"

if not exist "%venv_dir%" (
    python -m venv "%venv_dir%"
)

REM You may need to insyall microsoft c++ build tools (from visual studio, may be installed separately), windows 10/11 SDK, CUDA SDK

"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade pip
"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade -r "%script_dir%requirements.txt"
"%venv_dir%\Scripts\python" -m pip --require-virtualenv install --upgrade torch torchvision torchaudio --index-url=https://download.pytorch.org/whl/cu128

endlocal
