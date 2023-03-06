@echo off

if "%~2" == "" (
    echo Usage: %0 source_dir destination_dir
    exit /b 1
)

if not exist "%~1" (
    echo Error: %1 is not a directory
    exit /b 1
)

if not exist "%~2" mkdir "%~2"

xcopy /e /i /y "%~1\*" "%~2\"
