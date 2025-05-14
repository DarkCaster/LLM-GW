@echo off
setlocal

set "script_dir=%~dp0"
set "venv_dir=%script_dir%venv"

if not exist "%venv_dir%" (
    call "%script_dir%init.bat"
)

"%venv_dir%\Scripts\python" "%script_dir%download_hf_repo.py" %*

endlocal
