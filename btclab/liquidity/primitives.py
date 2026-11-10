"""Boolean primitives for H5 (OI/price divergence, leverage flush) and H6 (depth imbalance gate).

Separate from btclab.indicators.registry on purpose; same shape (REGISTRY, compute).
df: perp OHLCV (DatetimeIndex = bar open time). metrics: metrics_archive_5m frame;
depth_snap: output of depth.depth_snapshots. NaN inputs give False (A1.4), except the H6
*_ok gates, which are True (no information is no veto).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import depth as _depth
from . import liq_model as _lm
from . import volume_profile as _vp
from ._util import rolling_z


@dataclass(frozen=True)
class PrimSpec:
    name: str
    func: Callable
    defaults: dict
    space: dict
    primitives: tuple
    needs: tuple = ()
    description: str = ""


def _tfs(df, tf):
    return tf or str(df.index[1] - df.index[0])


def _h5(df, metrics, k=1, w=60, tau=2.0, tf=None):
    tf = _tfs(df, tf)
    oi = _lm.oi_on_bars(metrics, df.index, tf, "sum_open_interest")
    top = _lm.oi_on_bars(metrics, df.index, tf, "sum_toptrader_long_short_ratio")
    c = df["close"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        doi = np.full(len(c), np.nan)
        doi[k:] = np.log(oi[k:] / oi[:-k])
        r = np.full(len(c), np.nan)
        r[k:] = np.log(c[k:] / c[:-k])
    zo, zt = rolling_z(doi, w), rolling_z(top, w)
    with np.errstate(invalid="ignore"):
        return {
            "fragile_longs": (r > 0) & (zo > tau) & (zt > tau),
            "fragile_shorts": (r < 0) & (zo > tau) & (zt < -tau),
            "flush": (r < 0) & (zo < -tau),
        }


def _liq_flush(df, metrics, k=1, w=60, tau=2.0, range_mult=1.5, tf=None):
    tf = _tfs(df, tf)
    oi = _lm.oi_on_bars(metrics, df.index, tf)
    return {"flush": _lm.liq_flush(df["close"], df["high"], df["low"], oi, k, w, tau, range_mult)}


def _h6(df, depth_snap, band=1, w=40, gamma=1.0, tf=None):
    tf = _tfs(df, tf)
    imb = _depth.depth_imbalance(depth_snap, df.index, tf, band).to_numpy()
    z = rolling_z(imb, w)
    with np.errstate(invalid="ignore"):
        bids_thin = z < -gamma
        asks_thin = z > gamma
    return {"bids_thin": bids_thin, "asks_thin": asks_thin,
            "long_ok": ~bids_thin, "short_ok": ~asks_thin}


def _vprof(df, window=168, step=0.0025):
    p = _vp.rolling_profile(df["high"], df["low"], df["volume"], window, step)
    # shift by one bar: compare price to the PREVIOUS bar's value area
    vah = np.concatenate(([np.nan], p["vah"][:-1]))
    val = np.concatenate(([np.nan], p["val"][:-1]))
    c = df["close"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        return {"above_vah": c > vah, "below_val": c < val,
                "inside_value": (c <= vah) & (c >= val)}


def _f(lo, hi, st):
    return {"type": "float", "low": lo, "high": hi, "step": st}


def _i(lo, hi, st=1):
    return {"type": "int", "low": lo, "high": hi, "step": st}


REGISTRY: dict[str, PrimSpec] = {
    "oi_div": PrimSpec("oi_div", _h5, {"k": 1, "w": 60, "tau": 2.0},
                       {"k": {"type": "cat", "choices": [1, 3]}, "w": _i(30, 90, 10), "tau": _f(1.5, 3.0, 0.25)},
                       ("oi_div.fragile_longs", "oi_div.fragile_shorts", "oi_div.flush"), ("metrics",),
                       "H5 OI/price divergence and flush legs"),
    "liq_flush": PrimSpec("liq_flush", _liq_flush, {"k": 1, "w": 60, "tau": 2.0, "range_mult": 1.5},
                          {"tau": _f(1.5, 3.0, 0.25)}, ("liq_flush.flush",), ("metrics",),
                          "H5 flush: OI z drop with wide-range down bar"),
    "depth_imb": PrimSpec("depth_imb", _h6, {"band": 1, "w": 40, "gamma": 1.0},
                          {"band": {"type": "cat", "choices": [1, 2]}, "w": _i(20, 60, 10), "gamma": _f(0.5, 1.5, 0.25)},
                          ("depth_imb.bids_thin", "depth_imb.asks_thin", "depth_imb.long_ok", "depth_imb.short_ok"),
                          ("depth_snap",), "H6 depth imbalance z-gate (entries only)"),
    "vol_profile": PrimSpec("vol_profile", _vprof, {"window": 168, "step": 0.0025},
                            {"window": _i(72, 336, 24)},
                            ("vol_profile.above_vah", "vol_profile.below_val", "vol_profile.inside_value"), (),
                            "Price vs previous bar's rolling value area"),
}


def compute(name: str, df: pd.DataFrame, metrics=None, depth_snap=None, **params) -> dict:
    spec = REGISTRY[name]
    kw = {**spec.defaults, **params}
    extra = {"metrics": metrics, "depth_snap": depth_snap}
    out = spec.func(df, *[extra[n] for n in spec.needs], **kw)
    return {k: np.asarray(v, bool) for k, v in out.items()}


def all_primitives() -> list[str]:
    return [p for s in REGISTRY.values() for p in s.primitives]
