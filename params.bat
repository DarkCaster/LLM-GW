REM add local uv.exe path to the end, will prefer system uv.exe if present
set "PATH=%PATH%;%script_dir%tools"

REM UV settings
set "UV_CONCURRENT_DOWNLOADS=4"
set "UV_CONCURRENT_INSTALLS=2"

REM proxy server, UV will use it to download packages and python dist
REM set "ALL_PROXY=socks5://user:password@127.0.0.1:1080"

REM custom directory for pyc files
REM set "PYTHONPYCACHEPREFIX=%base_dir%cache"
