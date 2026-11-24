@echo off
rem One-time setup: creates a private Python environment and installs everything.
cd /d "%~dp0"
echo Setting up Tree Divider...
python --version >nul 2>&1 || (echo. & echo Python was not found. Install it from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
python -m venv venv || (echo Could not create the environment. & pause & exit /b 1)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-vision.txt -r requirements-auto.txt || (echo. & echo Installing failed - see the messages above. & pause & exit /b 1)
echo.
echo Done! Double-click "Tree Divider (Windows).bat" to start the app.
pause
