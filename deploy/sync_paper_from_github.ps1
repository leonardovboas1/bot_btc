# Pulls the GitHub-hosted paper ledger into data/paper so the local site shows it.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
git -C $root fetch -q origin paper-data
$dst = Join-Path $root "data\paper\paper.sqlite"
$tmp = "$dst.new"
cmd /c "git -C `"$root`" show FETCH_HEAD:paper.sqlite > `"$tmp`""
Remove-Item "$dst-wal", "$dst-shm" -ErrorAction SilentlyContinue
Move-Item -Force $tmp $dst
"paper ledger synced $(Get-Date -Format 'HH:mm:ss')"
