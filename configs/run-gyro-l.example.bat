@echo off
rem Starts the Strata server with Gyro-L. Edit ROOT and CONFIG; stop it by closing this window.
title Strata gyro-l
set "ROOT=C:\strata-gyro-l"
set "CONFIG=%ROOT%\configs\strata-gyro-l.json"
cd /d "%ROOT%\strata"
"%ROOT%\strata\.venv\Scripts\python.exe" "%ROOT%\strata\serve\server.py" --engine strata --config "%CONFIG%" --port 8098 --open
pause
