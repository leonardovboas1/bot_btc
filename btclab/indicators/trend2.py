"""Trend family 2: Ichimoku, Hull MA, SSL channel, UT Bot, Range Filter, QQE. All causal."""
from __future__ import annotations

import numpy as np
from numba import njit

from .base import atr, crossover, crossunder, ema, f64, highest, hma, lowest, rsi, shift, sma


# ----------------------------------------------------------------------------- Ichimoku
def ichimoku(high, low, close, conv: int = 9, base: int = 26, span_b: int = 52, disp: int = 26) -> dict:
    """Ichimoku. The cloud used for signals is the one *plotted at the current bar*, i.e.
    spans computed ``disp-1`` bars ago (Pine offset = disp-1). No chikou / forward cloud is
    used, so nothing depends on future bars."""
    h, l, c = f64(high), f64(low), f64(close)
    tenkan = (highest(h, conv) + lowest(l, conv)) / 2.0
    kijun = (highest(h, base) + lowest(l, base)) / 2.0
    sa = (tenkan + kijun) / 2.0
    sb = (highest(h, span_b) + lowest(l, span_b)) / 2.0
    ca, cb = shift(sa, disp - 1), shift(sb, disp - 1)
    top, bot = np.maximum(ca, cb), np.minimum(ca, cb)
    return {
        "tenkan": tenkan, "kijun": kijun, "cloud_a": ca, "cloud_b": cb,
        "cloud_top": top, "cloud_bottom": bot,
        "tk_cross_up": crossover(tenkan, kijun), "tk_cross_down": crossunder(tenkan, kijun),
        "above_cloud": c > top, "below_cloud": c < bot,
        "cloud_bull": ca > cb,
    }


# ----------------------------------------------------------------------------- Hull MA
def hull(close, length: int = 55) -> dict:
    c = f64(close)
    h = hma(c, length)
    h1, h2 = shift(h, 1), shift(h, 2)
    return {
        "hma": h, "rising": h > h1, "falling": h < h1,
        "turn_up": (h > h1) & (h1 <= h2), "turn_down": (h < h1) & (h1 >= h2),
        "close_cross_up": crossover(c, h), "close_cross_down": crossunder(c, h),
    }


# ----------------------------------------------------------------------------- SSL channel
@njit(cache=True)
def _ssl_k(close, shi, slo):
    n = len(close)
    hlv = np.full(n, np.nan)
    prev = 0.0
    for i in range(n):
        if np.isnan(shi[i]) or np.isnan(slo[i]):
            continue
        if close[i] > shi[i]:
            prev = 1.0
        elif close[i] < slo[i]:
            prev = -1.0
        hlv[i] = prev
    return hlv


def ssl_channel(high, low, close, length: int = 10) -> dict:
    h, l, c = f64(high), f64(low), f64(close)
    shi, slo = sma(h, length), sma(l, length)
    hlv = _ssl_k(c, shi, slo)
    down = np.where(hlv < 0, shi, slo)
    up = np.where(hlv < 0, slo, shi)
    down[np.isnan(hlv)] = np.nan
    up[np.isnan(hlv)] = np.nan
    h1 = shift(hlv)
    return {
        "ssl_up": up, "ssl_down": down, "hlv": hlv,
        "bull": hlv == 1.0, "bear": hlv == -1.0,
        "flip_up": (hlv == 1.0) & (h1 == -1.0), "flip_down": (hlv == -1.0) & (h1 == 1.0),
    }


# ----------------------------------------------------------------------------- UT Bot
@njit(cache=True)
def _utbot_k(src, nloss):
    n = len(src)
    stop = np.full(n, np.nan)
    pos = np.zeros(n)
    prev = 0.0  # nz(stop[1])
    for i in range(n):
        if np.isnan(nloss[i]):
            continue
        s1 = src[i - 1] if i > 0 else np.nan
        if src[i] > prev and s1 > prev:
            cur = max(prev, src[i] - nloss[i])
        elif src[i] < prev and s1 < prev:
            cur = min(prev, src[i] + nloss[i])
        elif src[i] > prev:
            cur = src[i] - nloss[i]
        else:
            cur = src[i] + nloss[i]
        stop[i] = cur
        p0 = pos[i - 1] if i > 0 else 0.0
        if s1 < prev and src[i] > prev:
            pos[i] = 1.0
        elif s1 > prev and src[i] < prev:
            pos[i] = -1.0
        else:
            pos[i] = p0
        prev = cur
    return stop, pos


def ut_bot(high, low, close, key: float = 1.0, atr_len: int = 10) -> dict:
    """UT Bot Alerts (QuantNomad): ATR trailing stop; buy = close crosses above the stop."""
    h, l, c = f64(high), f64(low), f64(close)
    stop, pos = _utbot_k(c, key * atr(h, l, c, atr_len))
    return {
        "stop": stop, "pos": pos,
        "buy": (c > stop) & crossover(c, stop), "sell": (c < stop) & crossover(stop, c),
        "long_pos": pos == 1.0, "short_pos": pos == -1.0,
    }


# ----------------------------------------------------------------------------- Range filter
@njit(cache=True)
def _rangefilter_k(x, r):
    n = len(x)
    f = np.full(n, np.nan)
    up = np.zeros(n)
    dn = np.zeros(n)
    prev = x[0]
    for i in range(n):
        if np.isnan(r[i]):
            f[i] = np.nan
            continue
        if x[i] > prev:
            cur = prev if x[i] - r[i] < prev else x[i] - r[i]
        else:
            cur = prev if x[i] + r[i] > prev else x[i] + r[i]
        f[i] = cur
        if i > 0 and not np.isnan(f[i - 1]):
            if cur > f[i - 1]:
                up[i] = up[i - 1] + 1
            elif cur < f[i - 1]:
                dn[i] = dn[i - 1] + 1
            else:
                up[i] = up[i - 1]
                dn[i] = dn[i - 1]
        prev = cur
    return f, up, dn


def range_filter(close, period: int = 100, mult: float = 3.0) -> dict:
    """Range Filter (DonovanWall). Seeded with the first close until the range is valid."""
    c = f64(close)
    avrng = ema(np.abs(c - shift(c)), period)
    rng = ema(avrng, period * 2 - 1) * mult
    f, up, dn = _rangefilter_k(c, rng)
    bull = (c > f) & (up > 0)
    bear = (c < f) & (dn > 0)
    return {
        "filter": f, "upper": f + rng, "lower": f - rng,
        "bull": bull, "bear": bear,
        "buy": bull & ~shift(bull), "sell": bear & ~shift(bear),
    }


# ----------------------------------------------------------------------------- QQE
@njit(cache=True)
def _qqe_k(rsima, dar):
    n = len(rsima)
    lb = np.full(n, np.nan)
    sb = np.full(n, np.nan)
    tr_ = np.ones(n)
    fast = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(rsima[i]) or np.isnan(dar[i]):
            continue
        nl = rsima[i] - dar[i]
        ns = rsima[i] + dar[i]
        if i == 0 or np.isnan(lb[i - 1]):
            lb[i], sb[i] = nl, ns
            continue
        r1 = rsima[i - 1]
        lb[i] = max(lb[i - 1], nl) if (r1 > lb[i - 1] and rsima[i] > lb[i - 1]) else nl
        sb[i] = min(sb[i - 1], ns) if (r1 < sb[i - 1] and rsima[i] < sb[i - 1]) else ns
        t0 = tr_[i - 1]
        if rsima[i] > sb[i - 1] and r1 <= sb[i - 1]:
            tr_[i] = 1.0
        elif rsima[i] < lb[i - 1] and r1 >= lb[i - 1]:
            tr_[i] = -1.0
        else:
            tr_[i] = t0
    for i in range(n):
        if not np.isnan(lb[i]):
            fast[i] = lb[i] if tr_[i] == 1.0 else sb[i]
    return fast, tr_


def qqe(close, rsi_len: int = 14, smooth: int = 5, factor: float = 4.238) -> dict:
    """QQE (Pine 'QQE' by Roman Ficek): fast trailing line of the smoothed RSI."""
    c = f64(close)
    rsima = ema(rsi(c, rsi_len), smooth)
    wild = rsi_len * 2 - 1
    atr_rsi = np.abs(shift(rsima) - rsima)
    dar = ema(ema(atr_rsi, wild), wild) * factor
    fast, trend = _qqe_k(rsima, dar)
    trend = np.where(np.isnan(fast), np.nan, trend)
    return {
        "rsi_ma": rsima, "fast": fast, "trend": trend,
        "cross_up": crossover(rsima, fast), "cross_down": crossunder(rsima, fast),
        "bull": rsima > fast, "bear": rsima < fast,
    }
