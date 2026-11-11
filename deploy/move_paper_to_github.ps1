# One-time: stops the local paper service and seeds the `paper-data` branch with the current ledger,
# after which .github/workflows/paper.yml keeps it running every ~30 min on GitHub.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$pf = Join-Path $root "data\paper\paper.pid"
if (Test-Path $pf) {
  $procId = [int](Get-Content $pf)
  $p = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
  if ($p -and $p.CommandLine -like "*src.live.paper*") { Stop-Process -Id $procId -Force; "stopped local paper ($procId)"; Start-Sleep 2 }
}
$db = Join-Path $root "data\paper\paper.sqlite"
& $py -c "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);c.execute('PRAGMA wal_checkpoint(TRUNCATE)');c.execute('PRAGMA journal_mode=DELETE');c.close()" $db
$tmp = Join-Path $env:TEMP "paper_seed"; Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory $tmp | Out-Null
Copy-Item $db, (Join-Path $root "data\paper\strategies.json") $tmp
$url = (git -C $root remote get-url origin)
git -C $tmp init -q -b paper-data; git -C $tmp add .; git -C $tmp commit -qm "seed ledger"
git -C $tmp push -qf $url paper-data
Set-Content (Join-Path $root "data\paper\REMOTE") "ledger hosted on branch paper-data (GitHub Actions)"
"seeded paper-data; the GitHub workflow takes over from here; local API now pulls it every 5 min"
