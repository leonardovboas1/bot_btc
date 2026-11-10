"""Validate raw parquet files: gaps, duplicates, OHLC sanity, zero-volume bars.

Usage: .venv/Scripts/python -m src.data.validate
"""
from __future__ import annotations

import re

import pandas as pd

from .download import RAW

FREQ = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def check(path) -> dict:
    df = pd.read_parquet(path)
    idx = df.index
    res = {"file": f"{path.parent.name}/{path.stem}", "rows": len(df), "start": idx.min(), "end": idx.max(),
           "dups": int(idx.duplicated().sum()), "sorted": bool(idx.is_monotonic_increasing),
           "gap_runs": "-", "missing": "-", "ohlc_bad": "-", "zero_vol": "-", "first_missing": None}
    m = re.search(r"_(5m|15m|1h|4h|1d)$", path.stem)
    if m:
        step = pd.Timedelta(FREQ[m.group(1)])
        miss = pd.date_range(idx.min(), idx.max(), freq=step).difference(idx)
        res["missing"] = len(miss)
        res["gap_runs"] = int(pd.Series(miss).diff().ne(step).sum()) if len(miss) else 0
        res["first_missing"] = miss[0] if len(miss) else None
    elif path.stem == "funding_rate":
        d = idx.to_series().diff().dropna()
        bad = d[d > pd.Timedelta(hours=8, minutes=5)]
        res["gap_runs"] = len(bad)
        res["missing"] = int(((bad / pd.Timedelta(hours=8)) - 1).round().sum())
        res["first_missing"] = bad.index[0] if len(bad) else None
    if {"open", "high", "low", "close"} <= set(df.columns):
        o = df[["open", "close"]]
        bad = ((df.low > o.min(axis=1)) | (df.high < o.max(axis=1)) | (df.low > df.high)
               | df[["open", "high", "low", "close"]].isna().any(axis=1) | (df.low <= 0))
        res["ohlc_bad"] = int(bad.sum())
        res["zero_vol"] = int((df.volume == 0).sum())
    return res


def main() -> None:
    t = pd.DataFrame([check(p) for p in sorted(RAW.rglob("*.parquet"))])
    for c in ("start", "end", "first_missing"):
        t[c] = pd.to_datetime(t[c], utc=True).dt.strftime("%Y-%m-%d %H:%M")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
