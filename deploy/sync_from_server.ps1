# Pulls the server's ledgers into this PC so the local site (API + web) shows the live 24/7 data.
# Usage:  .\deploy\sync_from_server.ps1 -Server ubuntu@1.2.3.4 -Key C:\Users\leona\.ssh\signallab.key
param([Parameter(Mandatory)][string]$Server, [string]$Key = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sshArgs = @("-o", "StrictHostKeyChecking=accept-new"); if ($Key) { $sshArgs += @("-i", $Key) }
# consistent snapshots on the server (safe while the services keep writing)
ssh @sshArgs $Server "cd bot_trade/data && sqlite3 paper/paper.sqlite '.backup /tmp/paper.snap' && sqlite3 raw/hyperliquid/hl.sqlite '.backup /tmp/hl.snap'"
$tmp = Join-Path $env:TEMP "signallab_sync"; New-Item -ItemType Directory -Force $tmp | Out-Null
scp @sshArgs "${Server}:/tmp/paper.snap" "${Server}:/tmp/hl.snap" $tmp
foreach ($pair in @(@("paper.snap", "data\paper\paper.sqlite"), @("hl.snap", "data\raw\hyperliquid\hl.sqlite"))) {
  $dst = Join-Path $root $pair[1]
  Remove-Item "$dst-wal", "$dst-shm" -ErrorAction SilentlyContinue
  Move-Item -Force (Join-Path $tmp $pair[0]) $dst
}
foreach ($v in "binance", "bybit", "okx") { scp @sshArgs -r "${Server}:bot_trade/data/raw/liquidations/$v" (Join-Path $root "data\raw\liquidations\") 2>$null }
"synced $(Get-Date -Format 'HH:mm:ss')"
