@echo off
rem Drag one or more tree photos (or a folder) onto this file to measure them automatically.
rem ---- settings you can change -------------------------------------------------------
set BANNER_M=1
set BANNER_WORDS=banner
rem   Describe what your banner looks like, e.g.:  white board / orange board / white sign
rem -------------------------------------------------------------------------------------
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Please run "Setup (Windows).bat" first. & pause & exit /b 1)
if "%~1"=="" (echo Drag tree photos onto this file to measure them. & pause & exit /b 1)
set OUT=%USERPROFILE%\Pictures\TreeDivider\Auto
echo Results will go to: %OUT%
echo (The first run downloads the models, about 335 MB - please wait.)
echo.
venv\Scripts\python.exe tools\auto_measure.py %* --banner-m %BANNER_M% --banner-words "%BANNER_WORDS%" --out "%OUT%" --txt "%OUT%\auto-readings.txt"
echo.
echo Done. Opening the results folder...
start "" "%OUT%"
pause
