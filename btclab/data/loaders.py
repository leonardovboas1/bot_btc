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


def _kline_path(market: str, tf: str) -> Path:
    if market not in _DIR:
        raise ValueError(f"unknown market {market!r}; use 'perp' or 'spot'")
    return RAW / _DIR[market] / f"klines_{tf}.parquet"


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
               allow_holdout: bool = False) -> pd.DataFrame:
    """OHLCV(+extras) indexed by UTC open_time; ``end`` is exclusive."""
    df = _slice(pd.read_parquet(_kline_path(market, tf)), start, end)
    _check(df.index.max() if len(df) else None, allow_holdout)
    return df


def load_funding(tf: str | None = None, start=None, end=None,
                 allow_holdout: bool = False) -> pd.Series:
    """Funding rate. With ``tf`` it is aligned to perp bars of that timeframe: a
    float Series on the bar index, non-zero on bars containing a funding event
    (rates inside one bar are summed). Without ``tf`` the raw event series."""
    raw = pd.read_parquet(RAW / "um_futures" / "funding_rate.parquet")["fundingRate"]
    if tf is None:
        raw = _slice(raw, start, end)
        _check(raw.index.max() if len(raw) else None, allow_holdout)
        return raw
    bars = load_ohlcv("perp", tf, start, end, allow_holdout)
    out = np.zeros(len(bars))
    if len(bars):
        step = pd.Timedelta(tf)
        pos = bars.index.searchsorted(raw.index, side="right") - 1
        ok = pos >= 0
        pos_c = np.clip(pos, 0, len(bars) - 1)
        ok &= raw.index.values < (bars.index.values[pos_c] + step.to_timedelta64())
        np.add.at(out, pos[ok], raw.values[ok])
    return pd.Series(out, index=bars.index, name="funding")
