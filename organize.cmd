@echo off

rem Environment Variables - Where the Python and Scripts are
setlocal
set PYTHONPATH=C:\Users\Jordan\PycharmProjects\oa-research;%PYTHONPATH%
set PYTH=C:\Users\Jordan\Documents\winpython\scripts\python.bat

rem Run Script
cmd -/k %PYTH% -m organize