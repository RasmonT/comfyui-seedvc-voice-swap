@echo off
rem Measurements for comfyui-seedvc-voice-swap. Installs nothing, downloads nothing, changes no existing file.
rem Reads its settings from tests\local.json (or pass the same options on the command line, see run_tests.py).
setlocal
set "TESTS=%~dp0"
set "PY=%TESTS%..\..\..\..\python_embeded\python.exe"
if not exist "%PY%" (
  echo ComfyUI embedded Python not found at "%PY%".
  echo Run manually:  path\to\python.exe "%TESTS%run_tests.py" --help
  pause
  exit /b 1
)
"%PY%" -s "%TESTS%run_tests.py" %*
echo.
echo Finished with exit code %ERRORLEVEL%. Results are in "%TESTS%results".
pause
