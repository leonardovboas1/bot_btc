"""Parquet loaders aligned to bars, with a persistent holdout lock.

The holdout cutoff is the last perp 1h bar minus 8 months. It is computed once,
stored in ``btclab/data/holdout.json`` and reused forever (never recomputed).
Returned data that reaches the cutoff raises unless ``allow_holdout=True`` or the
env var ``HOLDOUT_UNLOCK=1``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HOLDOUT_FILE = Path(__file__).with_name("holdout.json")
_DIR = {"perp": "um_futures", "spot": "spot"}
HOLDOUT_MONTHS = 8


class HoldoutError(RuntimeError):
    pass


def _base(symbol: str = "BTCUSDT") -> Path:
    """BTCUSDT keeps the legacy layout data/raw/{spot,um_futures}; others use data/raw/<SYMBOL>/."""
    symbol = symbol.upper()
    if not symbol.isalnum():
        raise ValueError(f"bad symbol {symbol!r}")
    return RAW if symbol == "BTCUSDT" else RAW / symbol


def _kline_path(market: str, tf: str, symbol: str = "BTCUSDT") -> Path:
    if market not in _DIR:
        raise ValueError(f"unknown market {market!r}; use 'perp' or 'spot'")
    return _base(symbol) / _DIR[market] / f"klines_{tf}.parquet"


def list_symbols() -> list[str]:
    """Symbols with perp 1h klines on disk (BTCUSDT first)."""
    out = []
    if (RAW / "um_futures" / "klines_1h.parquet").exists():
        out.append("BTCUSDT")
    if RAW.exists():
        out += sorted(d.name for d in RAW.iterdir()
                      if d.is_dir() and d.name not in _DIR.values()
                      and (d / "um_futures" / "klines_1h.parquet").exists())
    return out


def _resample_week(df: pd.DataFrame) -> pd.DataFrame:
    """1d -> 1w bars (weeks start Monday 00:00 UTC); partial last week is dropped."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for c in ("volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        if c in df.columns:
            agg[c] = "sum"
    g = df.resample("W-MON", label="left", closed="left")
    w = g.agg(agg)
    cnt = g["close"].count()
    w = w[cnt == 7]
    return w


def get_cutoff() -> pd.Timestamp:
    """Holdout cutoff (UTC). Bars with open_time >= cutoff are holdout."""
    if HOLDOUT_FILE.exists():
        return pd.Timestamp(json.loads(HOLDOUT_FILE.read_text())["cutoff"])
    src = _kline_path("perp", "1h")
    last = pd.read_parquet(src, columns=["close"]).index.max()
    cutoff = (last - pd.DateOffset(months=HOLDOUT_MONTHS)).floor("h")
    HOLDOUT_FILE.write_text(json.dumps({
        "cutoff": cutoff.isoformat(),
        "last_perp_bar": last.isoformat(),
        "months": HOLDOUT_MONTHS,
        "source": "klines_1h perp",
    }, indent=2))
    return cutoff


def _utc(x):
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _check(last_ts, allow_holdout: bool) -> None:
    if allow_holdout or os.environ.get("HOLDOUT_UNLOCK") == "1":
        return
    cutoff = get_cutoff()
    if last_ts is not None and last_ts >= cutoff:
        raise HoldoutError(
            f"data reaches {last_ts} >= holdout cutoff {cutoff}; pass end<=cutoff "
            "(exclusive) or allow_holdout=True / HOLDOUT_UNLOCK=1")


def _slice(df, start, end):
    s, e = _utc(start), _utc(end)
    if s is not None:
        df = df[df.index >= s]
    if e is not None:
        df = df[df.index < e]
    return df


def load_ohlcv(market: str = "perp", tf: str = "4h", start=None, end=None,
               allow_holdout: bool = False, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """OHLCV(+extras) indexed by UTC open_time; ``end`` is exclusive. tf "1w" resamples 1d."""
    if tf == "1w":
        df = _resample_week(pd.read_parquet(_kline_path(market, "1d", symbol)))
    else:
        df = pd.read_parquet(_kline_path(market, tf, symbol))
    df = _slice(df, start, end)
    _check(df.index.max() if len(df) else None, allow_holdout)
    return df


def load_funding(tf: str | None = None, start=None, end=None,
                 allow_holdout: bool = False, symbol: str = "BTCUSDT") -> pd.Series:
    """Funding rate. With ``tf`` it is aligned to perp bars of that timeframe: a
    float Series on the bar index, non-zero on bars containing a funding event
    (rates inside one bar are summed). Without ``tf`` the raw event series."""
    raw = pd.read_parquet(_base(symbol) / "um_futures" / "funding_rate.parquet")["fundingRate"]
    if tf is None:
        raw = _slice(raw, start, end)
        _check(raw.index.max() if len(raw) else None, allow_holdout)
        return raw
    bars = load_ohlcv("perp", tf, start, end, allow_holdout, symbol)
    out = np.zeros(len(bars))
    if len(bars):
        step = pd.Timedelta(days=7) if tf == "1w" else pd.Timedelta(tf)
        pos = bars.index.searchsorted(raw.index, side="right") - 1
        ok = pos >= 0
        pos_c = np.clip(pos, 0, len(bars) - 1)
        ok &= raw.index.values < (bars.index.values[pos_c] + step.to_timedelta64())
        np.add.at(out, pos[ok], raw.values[ok])
    return pd.Series(out, index=bars.index, name="funding")
