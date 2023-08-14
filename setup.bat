@echo off

setlocal

REM List of dependencies to install
set "dependencies=Scrapy matplotlib"

REM Path to the Python executable
set "python_executable=python"

REM Install each dependency
for %%d in (%dependencies%) do (
    echo Installing %%d...
    %python_executable% -m pip install %%d
    if %errorlevel% neq 0 (
        echo Failed to install %%d.
        goto :eof
    )
)

echo All dependencies installed successfully.
pause

:end