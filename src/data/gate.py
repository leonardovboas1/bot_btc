"""Gate.io USDT-futures contract_stats backfill (5m, API caps history at 180 days). No keys.

    .venv/Scripts/python -m src.data.gate [--symbols BTCUSDT ...]

-> data/raw/gate/<SYMBOL>/contract_stats_5m.parquet (UTC index = bar start), columns:
open_interest_usd, open_interest (contracts), mark_price, long_liq_usd, short_liq_usd,
long_liq_size, short_liq_size (contracts), long_liq_amount, short_liq_amount (base coin),
lsr_taker, lsr_account. long_liq = long positions liquidated. Idempotent: resumes from the last row.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "data" / "raw" / "gate"
URL = "https://api.gateio.ws/api/v4/futures/usdt/contract_stats"
BASES = ("BTC", "ETH", "SOL", "BNB", "NEAR", "FET", "ADA", "XRP")
SYMBOLS = tuple(b + "USDT" for b in BASES)
COLS = ["open_interest_usd", "open_interest", "mark_price", "long_liq_usd", "short_liq_usd",
        "long_liq_size", "short_liq_size", "long_liq_amount", "short_liq_amount",
        "lsr_taker", "lsr_account"]
STEP = 300
PAGE = 2000                 # API returns limit+1 rows
MAX_DAYS = 180
SLEEP = 0.4                 # polite spacing (public limit is far higher)


def contract(symbol: str) -> str:
    return symbol[:-4] + "_USDT"


def path_for(symbol: str, root: Path = GATE_DIR) -> Path:
    return Path(root) / symbol / "contract_stats_5m.parquet"


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = 0.0
    out = df[COLS].astype(float)
    out.index = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
    out.index.name = "ts"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _get(sess, params: dict, tries: int = 5):
    for i in range(tries):
        r = sess.get(URL, params=params, timeout=20)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gate: gave up {params}")


def download(symbol: str, root: Path = GATE_DIR, now: float | None = None, sess=None) -> pd.DataFrame:
    sess = sess or requests.Session()
    sess.headers["User-Agent"] = "bot_trade-research/1.0"
    now = int(now or time.time())
    p = path_for(symbol, root)
    old = pd.read_parquet(p) if p.exists() else rows_to_df([])
    floor = now - MAX_DAYS * 86400 + 3600
    cur = max(floor, int(old.index.max().timestamp()) + STEP if len(old) else floor)
    parts = [old]
    while cur < now:
        d = _get(sess, dict(contract=contract(symbol), interval="5m", limit=PAGE, **{"from": cur}))
        if not isinstance(d, list) or not d:
            break
        parts.append(rows_to_df(d))
        nxt = int(d[-1]["time"]) + STEP
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(SLEEP)
    df = pd.concat([x for x in parts if len(x)]) if any(len(x) for x in parts) else parts[0]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return df


def load(symbol: str, root: Path = GATE_DIR) -> pd.DataFrame | None:
    p = path_for(symbol, root)
    return pd.read_parquet(p) if p.exists() else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    for s in ap.parse_args().symbols:
        d = download(s)
        print(f"{s}: {len(d)} rows {d.index.min()} -> {d.index.max()}", flush=True)
