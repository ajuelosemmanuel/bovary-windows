@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "setup\friend_setup.py" goto extract
set "probe=import sys,struct;sys.exit(not(sys.version_info[:2]>=(3,11) and struct.calcsize('P')==8))"
set "localpython=%LocalAppData%\Programs\Python\Python313\python.exe"
if exist "%localpython%" (
  "%localpython%" -I -c "%probe%" >nul 2>&1
  if not errorlevel 1 goto installed
)
py -3 -I -c "%probe%" >nul 2>&1
if not errorlevel 1 goto pyready
python -I -c "%probe%" >nul 2>&1
if not errorlevel 1 goto pythonready
echo This kit needs 64-bit Python 3.11 or newer.
choice /C YN /M "Download verified Python 3.13 from python.org and install it for your account"
if errorlevel 2 goto end
echo Downloading the official 64-bit installer...
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $dir=New-Item -ItemType Directory -Path (Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())); $exe=Join-Path $dir.FullName 'python.exe'; try { (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.13.17/python-3.13.17-amd64.exe',$exe); if ((Get-FileHash -LiteralPath $exe).Hash -ne '5da012101ba2e17113a9a1cd901e9ce1441086f2b7ecb5de06ec94163899a190') { throw 'Python installer checksum mismatch' }; $p=Start-Process -FilePath $exe -ArgumentList '/quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 Include_doc=0 AssociateFiles=0 Shortcuts=0 PrependPath=0' -Wait -PassThru; if ($p.ExitCode) { throw ('Python installer exit '+$p.ExitCode) } } finally { Remove-Item -LiteralPath $exe -ErrorAction SilentlyContinue; $dir.Delete() }"
if errorlevel 1 goto manual
if exist "%localpython%" (
  "%localpython%" -I -c "%probe%" >nul 2>&1
  if not errorlevel 1 goto installed
)
:manual
echo Install the Windows 64-bit Python package from https://www.python.org/downloads/windows/
echo Then double-click Start Bovary again.
goto end
:extract
echo Right-click the downloaded ZIP, choose Extract All, and start Bovary from the extracted folder.
goto end
:installed
"%localpython%" -I setup\friend_setup.py
goto finished
:pyready
py -3 -I setup\friend_setup.py
goto finished
:pythonready
python -I setup\friend_setup.py
:finished
if not errorlevel 1 exit /b 0
:end
pause
exit /b 1
