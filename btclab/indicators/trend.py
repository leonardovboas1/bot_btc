"""Trend family: Supertrend, ADX/DMI, Parabolic SAR, MA crosses (Pine built-in semantics)."""
from __future__ import annotations

import numpy as np
from numba import njit

from .base import (_div, atr, change, crossover, crossunder, ema, f64, fixnan, hl2, hma, rma,
                   shift, sma, tr, wma)


# ----------------------------------------------------------------------------- Supertrend
@njit(cache=True)
def _supertrend_k(src, close, atrv, factor):
    n = len(src)
    st = np.full(n, np.nan)
    dirn = np.full(n, np.nan)
    L = np.zeros(n)
    U = np.zeros(n)
    for i in range(n):
        ub = src[i] + factor * atrv[i]
        lb = src[i] - factor * atrv[i]
        prevL = L[i - 1] if i > 0 else 0.0  # nz(lowerBand[1])
        prevU = U[i - 1] if i > 0 else 0.0  # nz(upperBand[1])
        if np.isnan(prevL):
            prevL = 0.0
        if np.isnan(prevU):
            prevU = 0.0
        c1 = close[i - 1] if i > 0 else np.nan
        L[i] = lb if (lb > prevL or c1 < prevL) else prevL
        U[i] = ub if (ub < prevU or c1 > prevU) else prevU
        if i == 0 or np.isnan(atrv[i - 1]):
            d = 1.0
        elif st[i - 1] == prevU:
            d = -1.0 if close[i] > U[i] else 1.0
        else:
            d = 1.0 if close[i] < L[i] else -1.0
        dirn[i] = d
        st[i] = L[i] if d == -1.0 else U[i]
    return st, dirn, L, U


def supertrend(high, low, close, atr_len: int = 10, factor: float = 3.0) -> dict:
    """ta.supertrend(factor, atrPeriod). direction: -1 = uptrend, +1 = downtrend (Pine)."""
    h, l, c = f64(high), f64(low), f64(close)
    a = atr(h, l, c, atr_len)
    st, d, _, _ = _supertrend_k(hl2(h, l), c, a, float(factor))
    warm = np.isnan(a)
    st = st.copy()
    st[warm] = np.nan  # Pine plots 0-bands during warm-up; masked here (does not feed back)
    d = d.copy()
    d[warm] = np.nan
    d1 = shift(d)
    return {
        "supertrend": st, "direction": d,
        "up": d == -1.0, "down": d == 1.0,
        "flip_up": (d == -1.0) & (d1 == 1.0),
        "flip_down": (d == 1.0) & (d1 == -1.0),
    }


# ----------------------------------------------------------------------------- DMI / ADX
def dmi(high, low, close, di_len: int = 14, adx_len: int = 14, adx_thr: float = 25.0) -> dict:
    """ta.dmi(diLength, adxSmoothing)."""
    h, l, c = f64(high), f64(low), f64(close)
    up = change(h)
    down = -change(l)
    plus_dm = np.where(np.isnan(up), np.nan, np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(np.isnan(down), np.nan, np.where((down > up) & (down > 0), down, 0.0))
    trur = rma(tr(h, l, c, False), di_len)
    plus = fixnan(100.0 * _div(rma(plus_dm, di_len), trur))
    minus = fixnan(100.0 * _div(rma(minus_dm, di_len), trur))
    s = plus + minus
    adx = 100.0 * rma(np.abs(plus - minus) / np.where(s == 0, 1.0, s), adx_len)
    strong = adx > adx_thr
    return {
        "plus_di": plus, "minus_di": minus, "adx": adx,
        "di_cross_up": crossover(plus, minus), "di_cross_down": crossunder(plus, minus),
        "bull": plus > minus, "bear": plus < minus,
        "adx_strong": strong, "adx_rising": adx > shift(adx),
        "bull_strong": (plus > minus) & strong, "bear_strong": (plus < minus) & strong,
    }


# ----------------------------------------------------------------------------- PSAR
@njit(cache=True)
def _sar_k(high, low, close, start, inc, mx):
    n = len(close)
    out = np.full(n, np.nan)
    result = np.nan
    maxmin = np.nan
    acc = np.nan
    below = False
    for i in range(1, n):
        first = False
        if i == 1:
            if close[1] > close[0]:
                below = True
                maxmin = high[1]
                result = low[0]
            else:
                below = False
                maxmin = low[1]
                result = high[0]
            first = True
            acc = start
        result = result + acc * (maxmin - result)
        if below:
            if result > low[i]:
                first = True
                below = False
                result = max(high[i], maxmin)
                maxmin = low[i]
                acc = start
        else:
            if result < high[i]:
                first = True
                below = True
                result = min(low[i], maxmin)
                maxmin = high[i]
                acc = start
        if not first:
            if below:
                if high[i] > maxmin:
                    maxmin = high[i]
                    acc = min(acc + inc, mx)
            else:
                if low[i] < maxmin:
                    maxmin = low[i]
                    acc = min(acc + inc, mx)
        if below:
            result = min(result, low[i - 1])
            if i > 1:
                result = min(result, low[i - 2])
        else:
            result = max(result, high[i - 1])
            if i > 1:
                result = max(result, high[i - 2])
        out[i] = result
    return out


def psar(high, low, close, start: float = 0.02, inc: float = 0.02, maximum: float = 0.2) -> dict:
    """ta.sar(start, inc, max) per the Pine reference implementation."""
    h, l, c = f64(high), f64(low), f64(close)
    s = _sar_k(h, l, c, float(start), float(inc), float(maximum))
    below = s < c
    b1 = shift(np.where(np.isnan(s), np.nan, below.astype(np.float64)))
    return {
        "sar": s,
        "long": below, "short": s > c,
        "flip_long": below & (b1 == 0.0),
        "flip_short": (s > c) & (b1 == 1.0),
    }


# ----------------------------------------------------------------------------- MA cross
_MA = {"ema": ema, "sma": sma, "wma": wma, "hma": hma, "rma": rma}


def ma_cross(close, fast: int = 20, slow: int = 50, kind: str = "ema") -> dict:
    """Fast/slow moving-average cross (EMA by default)."""
    c = f64(close)
    f = _MA[kind](c, fast)
    s = _MA[kind](c, slow)
    return {
        "fast": f, "slow": s,
        "fast_above_slow": f > s, "fast_below_slow": f < s,
        "cross_up": crossover(f, s), "cross_down": crossunder(f, s),
        "close_above_slow": c > s, "close_below_slow": c < s,
        "close_above_fast": c > f,
    }


ema_cross = ma_cross
