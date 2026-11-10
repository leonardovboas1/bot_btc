"""Volatility family: Bollinger, Keltner, TTM squeeze, Donchian, Chandelier exit. All causal."""
from __future__ import annotations

import numpy as np
from numba import njit

from .base import atr, crossover, crossunder, ema, f64, highest, lowest, shift, sma


@njit(cache=True)
def _stdev_k(x, n):
    """Population stdev (Pine ta.stdev, biased=true), NaN until a full valid window."""
    m = len(x)
    out = np.full(m, np.nan)
    for t in range(n - 1, m):
        s = 0.0
        ok = True
        for j in range(t - n + 1, t + 1):
            if np.isnan(x[j]):
                ok = False
                break
            s += x[j]
        if not ok:
            continue
        mu = s / n
        v = 0.0
        for j in range(t - n + 1, t + 1):
            v += (x[j] - mu) ** 2
        out[t] = np.sqrt(v / n)
    return out


@njit(cache=True)
def _linreg_k(y, n):
    """ta.linreg(y, n, 0): regression line value at the current bar."""
    m = len(y)
    out = np.full(m, np.nan)
    xm = (n - 1) / 2.0
    sxx = 0.0
    for i in range(n):
        sxx += (i - xm) ** 2
    for t in range(n - 1, m):
        ys = 0.0
        ok = True
        for j in range(n):
            v = y[t - n + 1 + j]
            if np.isnan(v):
                ok = False
                break
            ys += v
        if not ok:
            continue
        ym = ys / n
        sxy = 0.0
        for j in range(n):
            sxy += (j - xm) * (y[t - n + 1 + j] - ym)
        slope = sxy / sxx
        out[t] = ym + slope * (n - 1 - xm)
    return out


def stdev(src, length: int) -> np.ndarray:
    return _stdev_k(f64(src), int(length))


def linreg(src, length: int) -> np.ndarray:
    return _linreg_k(f64(src), int(length))


def bollinger(close, length: int = 20, mult: float = 2.0) -> dict:
    c = f64(close)
    basis = sma(c, length)
    d = mult * stdev(c, length)
    up, lo = basis + d, basis - d
    return {
        "basis": basis, "upper": up, "lower": lo,
        "cross_above_upper": crossover(c, up), "cross_below_lower": crossunder(c, lo),
        "cross_above_basis": crossover(c, basis), "cross_below_basis": crossunder(c, basis),
    }


def keltner(high, low, close, length: int = 20, mult: float = 1.5, atr_len: int = 10) -> dict:
    h, l, c = f64(high), f64(low), f64(close)
    basis = ema(c, length)
    rng = mult * atr(h, l, c, atr_len)
    up, lo = basis + rng, basis - rng
    return {
        "basis": basis, "upper": up, "lower": lo,
        "cross_above_upper": crossover(c, up), "cross_below_lower": crossunder(c, lo),
        "above_upper": c > up, "below_lower": c < lo,
    }


def ttm_squeeze(high, low, close, length: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5) -> dict:
    """TTM Squeeze (LazyBear): BB inside KC (KC uses SMA basis and SMA(TR) range)."""
    h, l, c = f64(high), f64(low), f64(close)
    basis = sma(c, length)
    dv = bb_mult * stdev(c, length)
    bbu, bbl = basis + dv, basis - dv
    c1 = shift(c)
    trv = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
    trv[0] = (h - l)[0]
    rng = sma(trv, length) * kc_mult
    kcu, kcl = basis + rng, basis - rng
    on = (bbl > kcl) & (bbu < kcu)
    off = (bbl < kcl) & (bbu > kcu)
    mid = (highest(h, length) + lowest(l, length)) / 2.0
    mom = linreg(c - (mid + basis) / 2.0, length)
    fire = shift(on) & ~on
    return {
        "momentum": mom, "squeeze_on": on, "squeeze_off": off, "fire": fire,
        "fire_long": fire & (mom > 0), "fire_short": fire & (mom < 0),
    }


def donchian(high, low, close, length: int = 20) -> dict:
    h, l, c = f64(high), f64(low), f64(close)
    up, lo = highest(h, length), lowest(l, length)
    up1, lo1 = shift(up), shift(lo)  # prior-bar channel: a breakout needs to exceed it
    mid = (up + lo) / 2.0
    return {
        "upper": up, "lower": lo, "mid": mid,
        "breakout_up": (c > up1) & ~(shift(c) > shift(up1)),
        "breakout_down": (c < lo1) & ~(shift(c) < shift(lo1)),
        "close_above_mid": c > mid, "close_below_mid": c < mid,
    }


@njit(cache=True)
def _chandelier_k(close, ls0, ss0):
    n = len(close)
    ls = ls0.copy()
    ss = ss0.copy()
    d = np.ones(n)
    for i in range(1, n):
        if close[i - 1] > ls[i - 1]:
            ls[i] = max(ls0[i], ls[i - 1])
        if close[i - 1] < ss[i - 1]:
            ss[i] = min(ss0[i], ss[i - 1])
        if close[i] > ss[i - 1]:
            d[i] = 1.0
        elif close[i] < ls[i - 1]:
            d[i] = -1.0
        else:
            d[i] = d[i - 1]
    return ls, ss, d


def chandelier(high, low, close, length: int = 22, mult: float = 3.0) -> dict:
    """Chandelier Exit (everget semantics). Bands masked to NaN during ATR warm-up."""
    h, l, c = f64(high), f64(low), f64(close)
    a = mult * atr(h, l, c, length)
    ls0 = highest(h, length) - a
    ss0 = lowest(l, length) + a
    # replace NaN warm-up with the raw value so recursion is well defined; mask after
    warm = np.isnan(ls0) | np.isnan(ss0)
    ls, ss, d = _chandelier_k(c, np.where(warm, -np.inf, ls0), np.where(warm, np.inf, ss0))
    ls[warm] = np.nan
    ss[warm] = np.nan
    d = d.copy()
    d[warm] = np.nan
    d1 = shift(d)
    return {
        "long_stop": ls, "short_stop": ss, "direction": d,
        "buy": (d == 1.0) & (d1 == -1.0), "sell": (d == -1.0) & (d1 == 1.0),
        "long_trend": d == 1.0, "short_trend": d == -1.0,
    }
