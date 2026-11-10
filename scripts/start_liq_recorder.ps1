# Starts the observed-liquidation recorder detached at Idle priority.
# Logs: data/raw/liquidations/recorder.log ; PID: data/raw/liquidations/recorder.pid
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "data\raw\liquidations"
New-Item -ItemType Directory -Force $logDir | Out-Null
$p = Start-Process -FilePath $py -ArgumentList "-m", "src.data.liq_recorder" -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardError (Join-Path $logDir "recorder.log") `
    -RedirectStandardOutput (Join-Path $logDir "recorder.out.log")
$p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Idle
Set-Content -Path (Join-Path $logDir "recorder.pid") -Value $p.Id
"liq_recorder started, PID $($p.Id), priority Idle"
