"""Momentum family: MACD, CCI, Williams %R, MFI (TradingView built-in semantics)."""
from __future__ import annotations

import numpy as np

from .base import (_div, change, crossover, crossunder, dev, ema, f64, highest, hlc3, lowest,
                   rolling_sum, sma)


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """ta.macd: macd = ema(fast) - ema(slow); signal = ema(macd, signal)."""
    c = f64(close)
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, signal)
    h = m - s
    up = crossover(m, s)
    dn = crossunder(m, s)
    return {
        "macd": m, "signal": s, "hist": h,
        "cross_up": up, "cross_down": dn,
        "cross_up_below_zero": up & (m < 0),
        "cross_down_above_zero": dn & (m > 0),
        "above_zero": m > 0, "below_zero": m < 0,
        "hist_rising": h > np.concatenate(([np.nan], h[:-1])),
        "zero_cross_up": crossover(m, 0.0), "zero_cross_down": crossunder(m, 0.0),
    }


def cci(high, low, close, length: int = 20, level: float = 100.0, src=None) -> dict:
    """ta.cci(src, length) = (src - sma) / (0.015 * dev); built-in default src = hlc3."""
    s = hlc3(high, low, close) if src is None else f64(src)
    v = _div(s - sma(s, length), 0.015 * dev(s, length))
    return {
        "cci": v,
        "cross_up_os": crossover(v, -level), "cross_down_ob": crossunder(v, level),
        "cross_up_ob": crossover(v, level), "cross_down_os": crossunder(v, -level),
        "above_zero": v > 0, "below_zero": v < 0,
    }


def williams_r(high, low, close, length: int = 14, os: float = -80.0, ob: float = -20.0) -> dict:
    """Williams %R = 100 * (close - highest(high)) / (highest(high) - lowest(low))."""
    hi = highest(high, length)
    lo = lowest(low, length)
    v = 100.0 * _div(f64(close) - hi, hi - lo)
    return {
        "wr": v,
        "cross_up_os": crossover(v, os), "cross_down_ob": crossunder(v, ob),
        "below_os": v < os, "above_ob": v > ob,
    }


def mfi(high, low, close, volume, length: int = 14, os: float = 20.0, ob: float = 80.0, src=None) -> dict:
    """ta.mfi(hlc3, length) per the Pine reference pseudo-code (bar 0, where change is na,
    contributes to both sums, as in Pine)."""
    s = hlc3(high, low, close) if src is None else f64(src)
    vol = f64(volume)
    ch = change(s)
    up_src = np.where(ch <= 0.0, 0.0, s)   # na <= 0 is False -> src
    dn_src = np.where(ch >= 0.0, 0.0, s)
    upper = rolling_sum(vol * up_src, length)
    lower = rolling_sum(vol * dn_src, length)
    v = 100.0 - 100.0 / (1.0 + _div(upper, lower))
    return {
        "mfi": v,
        "cross_up_os": crossover(v, os), "cross_down_ob": crossunder(v, ob),
        "below_os": v < os, "above_ob": v > ob,
    }
