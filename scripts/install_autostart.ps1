# Installs (or with -Remove, removes) the logon hook that runs scripts/autostart.ps1 hidden.
# Uses the per-user Startup folder: no admin rights needed.
param([switch]$Remove)
$startup = [Environment]::GetFolderPath("Startup")
$vbs = Join-Path $startup "SignalLab-autostart.vbs"
if ($Remove) { Remove-Item $vbs -ErrorAction SilentlyContinue; "removed $vbs"; return }
$ps1 = Join-Path $PSScriptRoot "autostart.ps1"
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """"$ps1"""""
Set-Content -Path $vbs -Encoding ASCII -Value "CreateObject(""WScript.Shell"").Run ""$cmd"", 0, False"
"installed $vbs"
Get-Content $vbs
