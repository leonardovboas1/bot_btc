# Starts the Hyperliquid exact-liquidation collector detached at Idle priority on cores 4-5.
# Logs: data/raw/hyperliquid/collector.log ; PID: data/raw/hyperliquid/collector.pid
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "data\raw\hyperliquid"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$p = Start-Process -FilePath $py -ArgumentList "-m", "src.data.hl_collector" -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardError (Join-Path $logDir "collector.log") `
    -RedirectStandardOutput (Join-Path $logDir "collector.out.log")
$p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Idle
$p.ProcessorAffinity = 48
Set-Content -Path (Join-Path $logDir "collector.pid") -Value $p.Id
"hl_collector started, PID $($p.Id), priority Idle, affinity 48"
