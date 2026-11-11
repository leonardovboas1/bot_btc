# Runs at Windows logon (Startup folder shortcut installed by scripts/install_autostart.ps1).
# 1) waits for internet  2) starts API (:8000), site (:5173), liq_recorder, hl_collector if not running
# 3) fills every data gap from while the PC was off (src.data.catchup), at Idle priority on 4 cores.
# Safe to run again by hand: anything already running is left alone.
# Logs: data/logs/autostart.log, api.log, web.log, catchup.log
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs "autostart.log"
function Log($m) { "$((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss'))Z $m" | Add-Content $log }
function Running($pattern) {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*$pattern*" }).Count -gt 0
}
function Listening($port) { @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue).Count -gt 0 }

Log "autostart begin"
$env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:NUMBA_NUM_THREADS = "1"

# 1) internet (up to ~5 min)
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-WebRequest "https://api.binance.com/api/v3/ping" -UseBasicParsing -TimeoutSec 5 | Out-Null; break }
    catch { Start-Sleep 5 }
}
Log "network check done after $($i*5)s"

# 2) services
if (-not (Listening 8000)) {
    $p = Start-Process $py -ArgumentList "-m", "btclab.api" -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardError (Join-Path $logs "api.log") -RedirectStandardOutput (Join-Path $logs "api.out.log")
    Log "API started PID $($p.Id)"
} else { Log "API already running" }

if (-not (Listening 5173)) {
    $web = Join-Path $root "web"
    $p = Start-Process "cmd.exe" -ArgumentList "/c", "npm run dev > `"$logs\web.log`" 2>&1" -WorkingDirectory $web -WindowStyle Hidden -PassThru
    Log "site started (npm run dev) PID $($p.Id)"
} else { Log "site already running" }

if (-not (Running "src.data.liq_recorder")) { Log (& (Join-Path $PSScriptRoot "start_liq_recorder.ps1")) } else { Log "liq_recorder already running" }
if (-not (Running "src.data.hl_collector")) { Log (& (Join-Path $PSScriptRoot "start_hl_collector.ps1")) } else { Log "hl_collector already running" }

# 3) catch-up (skip if one is already in progress)
if (-not (Running "src.data.catchup")) {
    $p = Start-Process $py -ArgumentList "-m", "src.data.catchup" -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardError (Join-Path $logs "catchup.err.log") -RedirectStandardOutput (Join-Path $logs "catchup.out.log")
    try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Idle; $p.ProcessorAffinity = 15 } catch {}
    Log "catch-up started PID $($p.Id)"
} else { Log "catch-up already running" }
Log "autostart end"
