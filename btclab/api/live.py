"""Live tail: bars after the last on-disk bar, fetched from Binance public REST (cached briefly).

Display only. Research code never calls this; backtests/signals keep using on-disk data.
"""
from __future__ import annotations

import threading
import time

import pandas as pd
import requests

_URL = {"perp": "https://fapi.binance.com/fapi/v1/klines",
        "spot": "https://api.binance.com/api/v3/klines"}
_LIMIT = {"perp": 1500, "spot": 1000}
_TTL = 15.0
_cache: dict = {}
_lock = threading.Lock()


def tail(market: str, tf: str, symbol: str, after: pd.Timestamp) -> pd.DataFrame:
    """Bars with open_time > ``after`` (UTC), last one may still be forming. Empty on any error."""
    key = (market, tf, symbol)
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit and now - hit[0] < _TTL:
        df = hit[1]
    else:
        try:
            r = requests.get(_URL[market], params={"symbol": symbol, "interval": tf,
                                                   "limit": _LIMIT[market]}, timeout=5)
            r.raise_for_status()
            rows = r.json()
        except Exception:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame([[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])]
                           for x in rows], columns=["t", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
        with _lock:
            _cache[key] = (now, df)
    a = pd.Timestamp(after)
    a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
    return df[df.index > a]
