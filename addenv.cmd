@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "USERPATH="
for /F "skip=2 tokens=1,2*" %%G in ('%SystemRoot%\System32\reg.exe query "HKCU\Environment" /v "Path" 2^>nul') do if /I "%%G" == "Path" (
    if /I "%%H" == "REG_EXPAND_SZ" (call set "USERPATH=%%I") else if /I "%%H" == "REG_SZ" set "USERPATH=%%I"
    if defined USERPATH goto :ADDENV
)
echo HKCU\Environment "Path" is not defined or has no string value.
goto :ENDBAT

:ADDENV
if "%1"=="" goto :ENDBAT
call set "PRUNED=%%USERPATH:%1=%%"
if not "%PRUNED%"=="%USERPATH%" goto :ENDBAT
for /F "skip=2 tokens=1,2*" %%G in ('%SystemRoot%\System32\reg.exe query "HKCU\Environment" /v "Path" 2^>nul') do if /I "%%G" == "Path" (
    echo %%H
    setx Path "%%I%1;"
)

:ENDBAT
endlocal
