@echo off
setlocal
rem Use Windows PowerShell modules even when launched through Python/cmd from pwsh 7.
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_course.ps1" %*
exit /b %ERRORLEVEL%
