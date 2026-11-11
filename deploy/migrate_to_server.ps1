# Moves the 24/7 services from this PC to the server, keeping the existing ledgers.
# Usage:  .\deploy\migrate_to_server.ps1 -Server ubuntu@1.2.3.4 -Key C:\Users\leona\.ssh\signallab.key
# Run deploy/setup.sh on the server first.
param([Parameter(Mandatory)][string]$Server, [string]$Key = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$sshArgs = @("-o", "StrictHostKeyChecking=accept-new"); if ($Key) { $sshArgs += @("-i", $Key) }

# 1) stop the local services (only processes whose command line matches the module)
$svc = @{ "data\paper\paper.pid" = "src.live.paper"; "data\raw\liquidations\recorder.pid" = "liq_recorder";
          "data\raw\hyperliquid\collector.pid" = "hl_collector" }
foreach ($f in $svc.Keys) {
  $pf = Join-Path $root $f
  if (Test-Path $pf) {
    $procId = [int](Get-Content $pf)
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
    if ($p -and $p.CommandLine -like "*$($svc[$f])*") { Stop-Process -Id $procId -Force; "stopped local $($svc[$f]) ($procId)" }
  }
}
Start-Sleep 2
# 2) fold the WAL into the main db files so a plain copy is consistent
& $py -c "import sqlite3,sys;[sqlite3.connect(p).execute('PRAGMA wal_checkpoint(TRUNCATE)') for p in sys.argv[1:]]" `
  (Join-Path $root "data\paper\paper.sqlite") (Join-Path $root "data\raw\hyperliquid\hl.sqlite")
# 3) upload ledgers, config and recorded liquidations
$remote = "bot_trade/data"
scp @sshArgs (Join-Path $root "data\paper\paper.sqlite") (Join-Path $root "data\paper\strategies.json") "${Server}:$remote/paper/"
scp @sshArgs (Join-Path $root "data\raw\hyperliquid\hl.sqlite") "${Server}:$remote/raw/hyperliquid/"
foreach ($v in "binance", "bybit", "okx") {
  $d = Join-Path $root "data\raw\liquidations\$v"
  if (Test-Path $d) { scp @sshArgs -r $d "${Server}:$remote/raw/liquidations/" }
}
# 4) start them on the server
ssh @sshArgs $Server "sudo systemctl enable --now signallab-paper signallab-liq signallab-hl && systemctl --no-pager --lines=0 status signallab-paper signallab-liq signallab-hl | grep -E 'signallab|Active'"
"Done. Local copies are refreshed by deploy\sync_from_server.ps1."
