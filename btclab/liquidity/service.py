"""FastAPI-ready, JSON-serializable entry points (no web imports here)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import depth, liq_model, volume_profile
from ._util import RAW


def _klines(tf: str, start, end) -> pd.DataFrame:
    k = pd.read_parquet(RAW / f"klines_{tf}.parquet")
    return k.loc[start:end] if (start or end) else k


def depth_heatmap_payload(start: str, end: str, tf: str = "1h") -> dict:
    snap = depth.depth_snapshots(depth.load_depth(start=start, end=end))
    k = _klines(tf, start, end)
    return depth.heatmap_json(depth.depth_heatmap(snap, k["close"], tf))


def depth_imbalance_payload(start: str, end: str, tf: str = "1h", band: int = 1) -> dict:
    snap = depth.depth_snapshots(depth.load_depth(start=start, end=end))
    k = _klines(tf, start, end)
    s = depth.depth_imbalance(snap, k.index, tf, band)
    return {"times": [t.isoformat() for t in s.index], "imbalance": [None if np.isnan(v) else float(v) for v in s]}


def liq_payload(start: str, end: str, tf: str = "1h", weights: dict | None = None) -> dict:
    k = _klines(tf, start, end)
    m = pd.read_parquet(RAW / "metrics_archive_5m.parquet")
    oi = liq_model.oi_on_bars(m, k.index, tf)
    res = liq_model.liq_levels(k["close"], k["high"], k["low"], oi, weights)
    flush = liq_model.liq_flush(k["close"], k["high"], k["low"], oi)
    out = liq_model.liq_heatmap_json(res, k.index)
    out["flush_times"] = [t.isoformat() for t in k.index[flush]]
    return out


def volume_profile_payload(start: str, end: str, tf: str = "1h") -> dict:
    k = _klines(tf, start, end)
    return volume_profile.profile_snapshot(k["high"], k["low"], k["volume"])
