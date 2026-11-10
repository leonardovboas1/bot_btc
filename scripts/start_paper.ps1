# Starts the forward paper-trading service detached at Idle priority on cores 4-5 (affinity mask 48).
# Logs: data/paper/paper.log ; PID: data/paper/paper.pid
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$dir = Join-Path $root "data\paper"
New-Item -ItemType Directory -Force $dir | Out-Null
$p = Start-Process -FilePath $py -ArgumentList "-m", "src.live.paper" -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $dir "paper.log") `
    -RedirectStandardError (Join-Path $dir "paper.err.log")
$p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Idle
$p.ProcessorAffinity = 48
Set-Content -Path (Join-Path $dir "paper.pid") -Value $p.Id
"paper service started, PID $($p.Id), priority Idle, affinity 48"
