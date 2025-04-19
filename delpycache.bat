@echo off
setlocal enabledelayedexpansion

:: Delete all __pycache__ folders in current directory and subdirectories
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        echo Deleting: %%d
        rmdir /s /q "%%d"
    )
)

echo All __pycache__ folders have been deleted.
pause