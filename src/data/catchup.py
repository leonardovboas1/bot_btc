"""Fill every data gap left while the PC was off. Run at startup by scripts/autostart.ps1.

    .venv/Scripts/python -m src.data.catchup [--skip-bookdepth]

Updates only datasets that already exist locally (per symbol folder under data/raw), each in its own
subprocess so one failure or a big file never takes the rest down. Order = what the site needs first:
klines -> funding -> metrics archive -> futdata (BTC, REST keeps ~30 days) -> Gate contract stats
(OI + observed liquidations, 180 days) -> BTC bookDepth (large, last).
Not recoverable after the fact: live websocket streams (Binance forceOrder in liq_recorder,
Hyperliquid collector) - they only resume from now on.
Log: data/logs/catchup.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
LOG = ROOT / "data" / "logs" / "catchup.log"
PY = sys.executable


def symbols() -> dict[str, Path]:
    out = {"BTCUSDT": RAW}  # BTC keeps legacy paths at data/raw/{um_futures,spot}
    for d in sorted(RAW.iterdir()):
        if d.is_dir() and d.name.endswith("USDT") and ((d / "um_futures").exists() or (d / "spot").exists()):
            out[d.name] = d
    return out


def present(base: Path, market: str, prefix: str) -> list[str]:
    folder = base / market
    return sorted(p.stem[len(prefix):] for p in folder.glob(f"{prefix}*.parquet")) if folder.exists() else []


def jobs(skip_bookdepth: bool) -> list[tuple[str, list[str]]]:
    syms = symbols()
    klines, other = [], []
    for s, base in syms.items():
        perp = present(base, "um_futures", "klines_")
        spot = present(base, "spot", "klines_")
        if perp:
            klines.append((f"{s} perp klines", ["klines", "--market", "perp", "--symbol", s, "--intervals", *perp]))
        if spot:
            klines.append((f"{s} spot klines", ["klines", "--market", "spot", "--symbol", s, "--intervals", *spot]))
        if (base / "um_futures" / "funding_rate.parquet").exists():
            other.append((f"{s} funding", ["funding", "--symbol", s]))
        if (base / "um_futures" / "metrics_archive_5m.parquet").exists():
            other.append((f"{s} metrics", ["metrics-archive", "--symbol", s, "--since", "2021-01-01"]))
    if (RAW / "um_futures" / "openInterestHist_5m.parquet").exists():
        other.append(("BTCUSDT futdata", ["futdata"]))
    out = [(n, [PY, "-m", "src.data.download", *a]) for n, a in klines + other]
    gate = sorted(p.name for p in (RAW / "gate").iterdir() if p.is_dir()) if (RAW / "gate").exists() else []
    if gate:
        out.append(("Gate contract stats", [PY, "-m", "src.data.gate", "--symbols", *gate]))
    if not skip_bookdepth and (RAW / "um_futures" / "bookdepth_archive.parquet").exists():
        out.append(("BTCUSDT bookDepth", [PY, "-m", "src.data.download", "bookdepth", "--since", "2023-01-01"]))
    return out


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bookdepth", action="store_true")
    a = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    todo = jobs(a.skip_bookdepth)
    log(f"catch-up start: {len(todo)} jobs")
    failed = []
    t0 = time.time()
    for name, cmd in todo:
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
        summary = "; ".join(lines[-6:])[:400]
        if r.returncode:
            failed.append(name)
            err = (r.stderr or "").strip().splitlines()[-1:] or ["?"]
            log(f"FAIL {name} ({time.time() - t:.0f}s): {err[0][:300]}")
        else:
            log(f"ok   {name} ({time.time() - t:.0f}s): {summary}")
    try:  # let a running API drop its caches so backtests/heatmaps see the new rows
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8000/admin/clear_cache", method="POST")
        log(f"API caches cleared: {urllib.request.urlopen(req, timeout=10).read().decode()}")
    except Exception as e:
        log(f"API cache clear skipped ({type(e).__name__})")
    log(f"catch-up done in {time.time() - t0:.0f}s, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
