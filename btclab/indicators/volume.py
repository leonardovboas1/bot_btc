"""Volume family: OBV, CMF, session VWAP, anchored VWAP, CVD approximation. All causal."""
from __future__ import annotations

import numpy as np
from numba import njit

from .base import _div, crossover, crossunder, f64, hlc3, rolling_sum, shift, sma

_DAY_MS = 86_400_000


def obv(close, volume, ma_len: int = 20) -> dict:
    c, v = f64(close), f64(volume)
    d = np.zeros(len(c))
    d[1:] = np.sign(np.diff(c)) * v[1:]
    o = np.cumsum(d)
    ma = sma(o, ma_len)
    return {"obv": o, "ma": ma, "cross_up_ma": crossover(o, ma), "cross_down_ma": crossunder(o, ma),
            "above_ma": o > ma}


def cmf(high, low, close, volume, length: int = 20) -> dict:
    h, l, c, v = f64(high), f64(low), f64(close), f64(volume)
    rng = h - l
    mfv = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng > 0, rng, 1.0) * v, 0.0)
    val = _div(rolling_sum(mfv, length), rolling_sum(v, length))
    return {"cmf": val, "above_zero": val > 0, "below_zero": val < 0,
            "cross_up_zero": crossover(val, 0.0), "cross_down_zero": crossunder(val, 0.0)}


@njit(cache=True)
def _vwap_k(pv, v, grp):
    n = len(pv)
    out = np.full(n, np.nan)
    spv = 0.0
    sv = 0.0
    cur = -1
    for i in range(n):
        if grp[i] < 0:
            continue
        if grp[i] != cur:
            cur = grp[i]
            spv = 0.0
            sv = 0.0
        spv += pv[i]
        sv += v[i]
        if sv > 0:
            out[i] = spv / sv
    return out


def _vwap_out(c, vw):
    return {"vwap": vw, "close_above": c > vw, "close_below": c < vw,
            "cross_up": crossover(c, vw), "cross_down": crossunder(c, vw)}


def vwap(high, low, close, volume, time) -> dict:
    """Session VWAP of hlc3, reset at 00:00 UTC (24/7 crypto session)."""
    h, l, c, v = f64(high), f64(low), f64(close), f64(volume)
    t = np.asarray(time, dtype=np.int64)
    grp = t // _DAY_MS
    vw = _vwap_k(hlc3(h, l, c) * v, v, grp)
    return _vwap_out(c, vw)


def anchored_vwap(high, low, close, volume, time, anchor_time: int = 0) -> dict:
    """VWAP accumulated from the first bar with open time >= anchor_time (epoch ms).
    The anchor is a user input (not detected from the future); 0 = start of the data."""
    h, l, c, v = f64(high), f64(low), f64(close), f64(volume)
    t = np.asarray(time, dtype=np.int64)
    grp = np.where(t >= int(anchor_time), 0, -1).astype(np.int64)
    vw = _vwap_k(hlc3(h, l, c) * v, v, grp)
    return _vwap_out(c, vw)


def cvd(open, high, low, close, volume, ma_len: int = 20) -> dict:
    """Cumulative volume delta APPROXIMATED from OHLCV: per-bar delta =
    volume * (2*close - high - low) / (high - low) (close location in the range).
    This is NOT true tick/taker-side CVD; it is a proxy."""
    h, l, c, v = f64(high), f64(low), f64(close), f64(volume)
    rng = h - l
    delta = np.where(rng > 0, (2 * c - h - l) / np.where(rng > 0, rng, 1.0) * v, 0.0)
    cum = np.cumsum(delta)
    ma = sma(cum, ma_len)
    return {"cvd": cum, "ma": ma, "cross_up_ma": crossover(cum, ma), "cross_down_ma": crossunder(cum, ma),
            "above_ma": cum > ma, "delta_pos": delta > 0}
