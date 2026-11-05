@echo off
rem Runs once, at the end of the unattended Windows install (dockurr/windows copies /oem to C:\OEM).
rem Installs Python 3.14.8 (hash-checked) and Windows-MCP 0.8.8, and starts Windows-MCP at every logon
rem in the interactive desktop session, which UI Automation needs.
rem Windows-MCP's HTTP transport has no authentication: the container forwards port 8000 only to the
rem lab's 127.0.0.1:8765, so reach it through an SSH tunnel and never publish it wider.
set LOG=C:\OEM\install.log
echo [%date% %time%] start >> %LOG%

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$u='https://www.python.org/ftp/python/3.14.8/python-3.14.8-amd64.exe'; $f='C:\OEM\python-3.14.8-amd64.exe';" ^
  "Invoke-WebRequest -UseBasicParsing $u -OutFile $f;" ^
  "$h=(Get-FileHash $f -Algorithm SHA256).Hash.ToLower();" ^
  "if ($h -ne '759be887b96e736a3ca886daf8d575f18fcae1a09efab6902f42d59e8999f8ef') { throw ('python installer hash mismatch: ' + $h) }" >> %LOG% 2>&1
if errorlevel 1 (echo [%date% %time%] python download failed >> %LOG% & exit /b 1)

C:\OEM\python-3.14.8-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 TargetDir=C:\Python314 >> %LOG% 2>&1
if not exist C:\Python314\python.exe (echo [%date% %time%] python install failed >> %LOG% & exit /b 1)

C:\Python314\python.exe -m pip install --no-warn-script-location windows-mcp==0.8.8 >> %LOG% 2>&1
if not exist C:\Python314\Scripts\windows-mcp.exe (echo [%date% %time%] windows-mcp install failed >> %LOG% & exit /b 1)

netsh advfirewall firewall add rule name="Windows-MCP 8000" dir=in action=allow protocol=TCP localport=8000 >> %LOG% 2>&1

rem Logon task for whoever logs on interactively (the autologon user), so the server sees that desktop.
schtasks /create /f /tn "Windows-MCP" /sc onlogon /rl highest /it /tr "C:\Python314\Scripts\windows-mcp.exe serve --transport streamable-http --host 0.0.0.0 --port 8000" >> %LOG% 2>&1

echo [%date% %time%] done >> %LOG%
exit /b 0
