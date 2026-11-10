"""Market structure: swing pivots, BOS/CHoCH, FVG, order blocks. Strictly causal:
a pivot is only known ``right`` bars after it forms, and every event is emitted at the
bar where it becomes knowable."""
from __future__ import annotations

import numpy as np
from numba import njit

from .base import f64, pivothigh, pivotlow


def swings(high, low, left: int = 5, right: int = 5) -> dict:
    """Pivots reported at the confirmation bar (pivot bar + right). ``last_high/low`` hold the
    most recent confirmed swing price (forward-filled)."""
    h, l = f64(high), f64(low)
    ph, pl = pivothigh(h, left, right), pivotlow(l, left, right)
    lh, ll = _ffill(ph), _ffill(pl)
    return {"pivot_high": ~np.isnan(ph), "pivot_low": ~np.isnan(pl),
            "pivot_high_price": ph, "pivot_low_price": pl, "last_high": lh, "last_low": ll}


@njit(cache=True)
def _ffill(x):
    out = np.full(len(x), np.nan)
    last = np.nan
    for i in range(len(x)):
        if not np.isnan(x[i]):
            last = x[i]
        out[i] = last
    return out


@njit(cache=True)
def _bos_k(close, ph, pl):
    n = len(close)
    bu = np.zeros(n, dtype=np.bool_)
    bd = np.zeros(n, dtype=np.bool_)
    cu = np.zeros(n, dtype=np.bool_)
    cd = np.zeros(n, dtype=np.bool_)
    hi = np.nan
    lo = np.nan
    trend = 0
    for i in range(n):
        if not np.isnan(ph[i]):
            hi = ph[i]
        if not np.isnan(pl[i]):
            lo = pl[i]
        if not np.isnan(hi) and close[i] > hi:
            if trend == -1:
                cu[i] = True
            else:
                bu[i] = True
            trend = 1
            hi = np.nan  # level consumed
        elif not np.isnan(lo) and close[i] < lo:
            if trend == 1:
                cd[i] = True
            else:
                bd[i] = True
            trend = -1
            lo = np.nan
    return bu, bd, cu, cd


def bos_choch(high, low, close, left: int = 5, right: int = 5) -> dict:
    """Break of structure (continuation) / change of character (reversal): a close beyond the
    latest *confirmed* swing level. CHoCH = break against the prevailing trend."""
    h, l, c = f64(high), f64(low), f64(close)
    bu, bd, cu, cd = _bos_k(c, pivothigh(h, left, right), pivotlow(l, left, right))
    return {"bos_up": bu, "bos_down": bd, "choch_up": cu, "choch_down": cd,
            "break_up": bu | cu, "break_down": bd | cd}


@njit(cache=True)
def _fvg_k(high, low, close, min_gap):
    n = len(close)
    fb = np.zeros(n, dtype=np.bool_)
    fs = np.zeros(n, dtype=np.bool_)
    rb = np.zeros(n, dtype=np.bool_)
    rs = np.zeros(n, dtype=np.bool_)
    bt, bb = np.nan, np.nan   # active bull gap top/bottom
    st, sb = np.nan, np.nan   # active bear gap top/bottom
    for i in range(n):
        # retest / mitigation of gaps formed on earlier bars
        if not np.isnan(bt):
            if close[i] < bb:
                bt = np.nan
            elif low[i] <= bt:
                rb[i] = True
        if not np.isnan(st):
            if close[i] > st:
                st = np.nan
            elif high[i] >= sb:
                rs[i] = True
        if i >= 2:
            if low[i] > high[i - 2] and (low[i] - high[i - 2]) >= min_gap * close[i]:
                fb[i] = True
                bt, bb = low[i], high[i - 2]
            if high[i] < low[i - 2] and (low[i - 2] - high[i]) >= min_gap * close[i]:
                fs[i] = True
                st, sb = low[i - 2], high[i]
    return fb, fs, rb, rs


def fvg(high, low, close, min_gap: float = 0.0) -> dict:
    """Fair value gaps (3-candle imbalance), known at the close of the third candle.
    ``retest_*``: a later bar trades back into the most recent un-mitigated gap
    (gap mitigated once price closes through it). ``min_gap`` is a fraction of close."""
    fb, fs, rb, rs = _fvg_k(f64(high), f64(low), f64(close), float(min_gap))
    return {"bull_fvg": fb, "bear_fvg": fs, "retest_bull": rb, "retest_bear": rs}


@njit(cache=True)
def _ob_k(open_, high, low, close, brk_up, brk_dn, lookback):
    n = len(close)
    fb = np.zeros(n, dtype=np.bool_)
    fs = np.zeros(n, dtype=np.bool_)
    rb = np.zeros(n, dtype=np.bool_)
    rs = np.zeros(n, dtype=np.bool_)
    bt, bb = np.nan, np.nan
    st, sb = np.nan, np.nan
    for i in range(n):
        if not np.isnan(bt):
            if close[i] < bb:
                bt = np.nan
            elif low[i] <= bt:
                rb[i] = True
        if not np.isnan(st):
            if close[i] > st:
                st = np.nan
            elif high[i] >= sb:
                rs[i] = True
        if brk_up[i]:  # last bearish candle before the break
            for j in range(i - 1, max(i - 1 - lookback, -1), -1):
                if close[j] < open_[j]:
                    bt, bb = high[j], low[j]
                    fb[i] = True
                    break
        if brk_dn[i]:
            for j in range(i - 1, max(i - 1 - lookback, -1), -1):
                if close[j] > open_[j]:
                    st, sb = high[j], low[j]
                    fs[i] = True
                    break
    return fb, fs, rb, rs


def order_blocks(open, high, low, close, left: int = 5, right: int = 5, lookback: int = 20) -> dict:
    """Order block = last opposite-colour candle before a structure break (BOS/CHoCH on
    confirmed swings). Known at the break bar; ``retest_*`` = later bar trades back into the
    zone before the zone is mitigated (close through it)."""
    o, h, l, c = f64(open), f64(high), f64(low), f64(close)
    s = bos_choch(h, l, c, left, right)
    fb, fs, rb, rs = _ob_k(o, h, l, c, s["break_up"], s["break_down"], int(lookback))
    return {"bull_ob": fb, "bear_ob": fs, "retest_bull": rb, "retest_bear": rs}
