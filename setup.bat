@echo off

REM Set the path to the virtual environment's activate script
set VENV_PATH=webui\venv
set SCRIPTS_PATH=%VENV_PATH%\Scripts

REM Check if Python 3.9 is available
set PYTHON_VER=python3.9
where %PYTHON_VER%
if %ERRORLEVEL% NEQ 0 (
    echo %PYTHON_VER% not found. Please install Python 3.9.
    exit /b 1
)

REM Check if the virtual environment exists, create if not
if not exist %SCRIPTS_PATH% (
    echo Creating virtual environment...
    python3.9 -m venv %VENV_PATH%
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created successfully.
)

REM Activate the virtual environment
call "%SCRIPTS_PATH%\activate"

REM Verify the Python version in the virtual environment
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Failed to verify Python version.
    exit /b 1
)

setlocal enabledelayedexpansion
for /f "tokens=2 delims= " %%i in ('python --version') do set PYTHON_VER_ACTUAL=%%i
echo Using Python version !PYTHON_VER_ACTUAL!
if not "!PYTHON_VER_ACTUAL!"=="3.9" (
    echo Warning: The virtual environment is not using Python 3.9.
)

REM Install the requirements from requirements.txt
pip install -r webui\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements.
    exit /b 1
)
endlocal

echo Setup complete. The virtual environment is activated, and requirements are installed.
pause