"""Pine Script v5/v6 ``ta.*`` semantics in numpy + numba.

Conventions
-----------
* Every function is pure: numpy arrays in, ``float64`` arrays (or ``bool``
  arrays for conditions) of the same length out.
* ``NaN`` plays the role of Pine's ``na``. Arithmetic with ``NaN`` propagates;
  comparisons with ``NaN`` are ``False`` (as in Pine, where ``na > x`` is false).
* No lookahead: value ``t`` only depends on inputs ``0..t``.
* Division by zero returns ``NaN`` (Pine returns ``na``), except where the
  Pine built-in special-cases it (``ta.rsi``).

Recursive filters (ema / rma / ...) follow the Pine reference pseudo-code
``sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * sum[1]``:
the filter is seeded with the SMA of the first ``length`` valid values, a
``NaN`` input yields ``NaN`` and the filter re-seeds afterwards.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

__all__ = [
    "f64", "shift", "nz", "hl2", "hlc3", "ohlc4",
    "sma", "ema", "rma", "wma", "hma", "rsi", "stoch", "tr", "atr",
    "highest", "lowest", "change", "rolling_sum", "dev", "fixnan",
    "crossover", "crossunder", "cross", "valuewhen", "pivothigh", "pivotlow",
    "heikinashi", "request_security_htf", "parse_tf", "times_ms",
]


# --------------------------------------------------------------------------- utils
def f64(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def shift(x, k: int = 1) -> np.ndarray:
    """Pine history operator ``x[k]`` (k >= 0). Bool arrays are padded with False."""
    x = np.asarray(x)
    out = np.empty_like(x) if x.dtype == bool else np.empty(len(x), dtype=np.float64)
    if x.dtype == bool:
        out[:] = False
    else:
        out[:] = np.nan
    if k == 0:
        out[:] = x
    elif k < len(x):
        out[k:] = x[:-k]
    return out


def nz(x, replacement: float = 0.0) -> np.ndarray:
    x = f64(x)
    return np.where(np.isnan(x), replacement, x)


def hl2(high, low) -> np.ndarray:
    return (f64(high) + f64(low)) / 2.0


def hlc3(high, low, close) -> np.ndarray:
    return (f64(high) + f64(low) + f64(close)) / 3.0


def ohlc4(open_, high, low, close) -> np.ndarray:
    return (f64(open_) + f64(high) + f64(low) + f64(close)) / 4.0


def _div(a, b) -> np.ndarray:
    """a / b with Pine semantics: division by zero -> na."""
    a = f64(a)
    b = f64(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[b == 0] = np.nan
    return out


# --------------------------------------------------------------------------- kernels
@njit(cache=True)
def _sma_k(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    if n < 1:
        return out
    for i in range(n - 1, m):
        s = 0.0
        ok = True
        for j in range(i - n + 1, i + 1):
            v = x[j]
            if np.isnan(v):
                ok = False
                break
            s += v
        if ok:
            out[i] = s / n
    return out


@njit(cache=True)
def _sum_k(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        s = 0.0
        ok = True
        for j in range(i - n + 1, i + 1):
            v = x[j]
            if np.isnan(v):
                ok = False
                break
            s += v
        if ok:
            out[i] = s
    return out


@njit(cache=True)
def _exp_filter_k(x, n, alpha):
    """Pine ema/rma: seeded by SMA(n) whenever the previous value is na."""
    m = len(x)
    out = np.full(m, np.nan)
    seed = _sma_k(x, n)
    prev = np.nan
    for i in range(m):
        if np.isnan(prev):
            cur = seed[i]
        else:
            cur = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = cur
        prev = cur
    return out


@njit(cache=True)
def _wma_k(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    norm = n * (n + 1) / 2.0
    for i in range(n - 1, m):
        s = 0.0
        ok = True
        for k in range(n):
            v = x[i - k]
            if np.isnan(v):
                ok = False
                break
            s += v * (n - k)
        if ok:
            out[i] = s / norm
    return out


@njit(cache=True)
def _extreme_k(x, n, is_max):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        best = x[i]
        ok = not np.isnan(best)
        if ok:
            for j in range(i - n + 1, i):
                v = x[j]
                if np.isnan(v):
                    ok = False
                    break
                if is_max:
                    if v > best:
                        best = v
                else:
                    if v < best:
                        best = v
        if ok:
            out[i] = best
    return out


@njit(cache=True)
def _dev_k(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    mean = _sma_k(x, n)
    for i in range(n - 1, m):
        mu = mean[i]
        if np.isnan(mu):
            continue
        s = 0.0
        for j in range(n):
            s += abs(x[i - j] - mu)
        out[i] = s / n
    return out


@njit(cache=True)
def _valuewhen_k(cond, src, occurrence):
    m = len(cond)
    out = np.full(m, np.nan)
    buf = np.full(occurrence + 1, np.nan)  # buf[0] = most recent
    count = 0
    for i in range(m):
        if cond[i]:
            for k in range(occurrence, 0, -1):
                buf[k] = buf[k - 1]
            buf[0] = src[i]
            count += 1
        if count > occurrence:
            out[i] = buf[occurrence]
    return out


@njit(cache=True)
def _pivot_k(x, left, right, is_high):
    m = len(x)
    out = np.full(m, np.nan)
    for t in range(left + right, m):
        p = t - right
        v = x[p]
        if np.isnan(v):
            continue
        ok = True
        for j in range(p - left, t + 1):
            if j == p:
                continue
            w = x[j]
            if np.isnan(w):
                ok = False
                break
            if is_high:
                if not (v > w):
                    ok = False
                    break
            else:
                if not (v < w):
                    ok = False
                    break
        if ok:
            out[t] = v
    return out


@njit(cache=True)
def _fixnan_k(x):
    out = x.copy()
    last = np.nan
    for i in range(len(x)):
        if np.isnan(x[i]):
            out[i] = last
        else:
            last = x[i]
    return out


@njit(cache=True)
def _ha_k(o, h, l, c):
    m = len(o)
    ho = np.full(m, np.nan)
    hc = (o + h + l + c) / 4.0
    hh = np.full(m, np.nan)
    hl = np.full(m, np.nan)
    for i in range(m):
        if i == 0 or np.isnan(ho[i - 1]):
            ho[i] = (o[i] + c[i]) / 2.0
        else:
            ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
        hh[i] = max(h[i], max(ho[i], hc[i]))
        hl[i] = min(l[i], min(ho[i], hc[i]))
    return ho, hh, hl, hc


# --------------------------------------------------------------------------- ta.*
def sma(src, length: int) -> np.ndarray:
    """ta.sma — na if any value in the window is na."""
    return _sma_k(f64(src), int(length))


def rolling_sum(src, length: int) -> np.ndarray:
    """math.sum."""
    return _sum_k(f64(src), int(length))


def ema(src, length: int) -> np.ndarray:
    """ta.ema — alpha = 2/(n+1), seeded with SMA(n) of the first n values."""
    n = int(length)
    return _exp_filter_k(f64(src), n, 2.0 / (n + 1.0))


def rma(src, length: int) -> np.ndarray:
    """ta.rma (Wilder) — alpha = 1/n, SMA seed."""
    n = int(length)
    return _exp_filter_k(f64(src), n, 1.0 / n)


def wma(src, length: int) -> np.ndarray:
    """ta.wma — linear weights n..1."""
    return _wma_k(f64(src), int(length))


def hma(src, length: int) -> np.ndarray:
    """ta.hma = wma(2*wma(src, n/2) - wma(src, n), round(sqrt(n))).

    Fidelity note: n/2 is truncated to int and sqrt(n) is rounded (half up)."""
    n = int(length)
    half = max(int(n // 2), 1)
    sq = max(int(math.floor(math.sqrt(n) + 0.5)), 1)
    return wma(2.0 * wma(src, half) - wma(src, n), sq)


def change(src, length: int = 1) -> np.ndarray:
    """ta.change = src - src[length]."""
    x = f64(src)
    return x - shift(x, int(length))


def rsi(src, length: int) -> np.ndarray:
    """ta.rsi: rma of up/down moves; down == 0 -> 100, up == 0 -> 0."""
    x = f64(src)
    ch = change(x)
    u = np.maximum(ch, 0.0)  # NaN propagates at bar 0
    d = np.maximum(-ch, 0.0)
    up = rma(u, length)
    dn = rma(d, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = 100.0 - 100.0 / (1.0 + up / dn)
    res = np.where(up == 0.0, 0.0, res)
    res = np.where(dn == 0.0, 100.0, res)
    res[np.isnan(up) | np.isnan(dn)] = np.nan
    return res


def highest(src, length: int) -> np.ndarray:
    """ta.highest(src, length) (window includes the current bar)."""
    return _extreme_k(f64(src), int(length), True)


def lowest(src, length: int) -> np.ndarray:
    return _extreme_k(f64(src), int(length), False)


def stoch(src, high, low, length: int) -> np.ndarray:
    """ta.stoch = 100 * (src - lowest(low)) / (highest(high) - lowest(low))."""
    lo = lowest(low, length)
    hi = highest(high, length)
    return 100.0 * _div(f64(src) - lo, hi - lo)


def tr(high, low, close, handle_na: bool = False) -> np.ndarray:
    """ta.tr(handle_na). At bar 0 (close[1] is na) -> high-low if handle_na else na."""
    h, l, c = f64(high), f64(low), f64(close)
    c1 = shift(c, 1)
    out = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
    first = np.isnan(c1)
    out[first] = (h - l)[first] if handle_na else np.nan
    return out


def atr(high, low, close, length: int) -> np.ndarray:
    """ta.atr = rma(tr(true), length)."""
    return rma(tr(high, low, close, True), length)


def dev(src, length: int) -> np.ndarray:
    """ta.dev — mean absolute deviation from the SMA."""
    return _dev_k(f64(src), int(length))


def fixnan(src) -> np.ndarray:
    return _fixnan_k(f64(src))


def _as_bool_inputs(a, b):
    a = f64(a)
    b = f64(b) if np.ndim(b) else np.full(len(a), float(b))
    return a, b


def crossover(a, b) -> np.ndarray:
    """ta.crossover: a > b and a[1] <= b[1]. ``b`` may be a scalar."""
    a, b = _as_bool_inputs(a, b)
    return (a > b) & (shift(a) <= shift(b))


def crossunder(a, b) -> np.ndarray:
    """ta.crossunder: a < b and a[1] >= b[1]."""
    a, b = _as_bool_inputs(a, b)
    return (a < b) & (shift(a) >= shift(b))


def cross(a, b) -> np.ndarray:
    return crossover(a, b) | crossunder(a, b)


def _truthy(cond) -> np.ndarray:
    """Pine bool coercion: for float series na and 0 are false."""
    c = np.asarray(cond)
    if c.dtype == bool:
        return c
    c = f64(c)
    return ~np.isnan(c) & (c != 0.0)


def valuewhen(cond, src, occurrence: int = 0) -> np.ndarray:
    """ta.valuewhen: src at the ``occurrence``-th most recent bar where cond was true
    (0 = latest, current bar included)."""
    return _valuewhen_k(_truthy(cond), f64(src), int(occurrence))


def pivothigh(src, left: int, right: int) -> np.ndarray:
    """ta.pivothigh — value reported at the confirmation bar t (pivot at t-right),
    NOT shifted back. Strict inequality vs. all neighbours (see fidelity notes)."""
    return _pivot_k(f64(src), int(left), int(right), True)


def pivotlow(src, left: int, right: int) -> np.ndarray:
    return _pivot_k(f64(src), int(left), int(right), False)


def heikinashi(open_, high, low, close):
    """Heikin Ashi candles (TradingView ``heikinashi()`` ticker).
    First bar: ha_open = (open + close) / 2. Returns (ha_open, ha_high, ha_low, ha_close)."""
    return _ha_k(f64(open_), f64(high), f64(low), f64(close))


# --------------------------------------------------------------------------- HTF
_MIN = 60_000
_DAY = 86_400_000
_WEEK = 7 * _DAY
_MONDAY_OFFSET = 4 * _DAY  # 1970-01-05 was a Monday


def parse_tf(tf) -> tuple[str, int]:
    """Parse a TradingView / pandas-ish timeframe.

    Returns (kind, value): kind 'ms' -> fixed duration in ms, 'W' -> n weeks, 'M' -> n months.
    Accepts '60', '240', '720', 'D', '1D', 'W', '1W', 'M', '3M', '4h', '15m', '1d', '1w', or int minutes.
    Uppercase 'M' is month (TradingView); lowercase 'm' / 'min' is minutes.
    """
    if isinstance(tf, (int, np.integer)):
        return "ms", int(tf) * _MIN
    s = str(tf).strip()
    if s.isdigit():
        return "ms", int(s) * _MIN
    num = "".join(ch for ch in s if ch.isdigit())
    unit = s[len(num):] if s.startswith(num) else s
    n = int(num) if num else 1
    if unit == "M":
        return "M", n
    u = unit.lower()
    if u in ("m", "min", "t"):
        return "ms", n * _MIN
    if u in ("h", "hr", "h1"):
        return "ms", n * 60 * _MIN
    if u in ("d", "day"):
        return "ms", n * _DAY
    if u in ("w", "wk", "week"):
        return "W", n
    if u in ("s",):
        return "ms", n * 1000
    raise ValueError(f"unrecognised timeframe {tf!r}")


def tf_ms(tf) -> int:
    kind, v = parse_tf(tf)
    if kind == "ms":
        return v
    if kind == "W":
        return v * _WEEK
    return v * 31 * _DAY  # nominal, only used for comparisons


def times_ms(obj) -> np.ndarray:
    """int64 epoch-ms open times from a DataFrame (DatetimeIndex or 'open_time'/'time'
    column), a DatetimeIndex, or an integer (ms) array."""
    if isinstance(obj, pd.DataFrame):
        for col in ("open_time", "time", "timestamp"):
            if col in obj.columns:
                return times_ms(obj[col].to_numpy())
        return times_ms(obj.index)
    if isinstance(obj, (pd.DatetimeIndex, pd.Series)) or (
        isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.datetime64)
    ):
        idx = pd.DatetimeIndex(obj)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx.as_unit("ms").asi8.astype(np.int64)
    arr = np.asarray(obj)
    if arr.dtype == object:
        return times_ms(pd.DatetimeIndex(arr))
    return arr.astype(np.int64)


def _bucket_start(t: np.ndarray, kind: str, v: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bucket_open_ms, bucket_close_ms) for each base timestamp."""
    if kind == "ms":
        start = (t // v) * v
        return start, start + v
    if kind == "W":
        span = v * _WEEK
        start = ((t - _MONDAY_OFFSET) // span) * span + _MONDAY_OFFSET
        return start, start + span
    months = t.astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)
    mstart = (months // v) * v
    start = mstart.astype("datetime64[M]").astype("datetime64[ms]").astype(np.int64)
    end = (mstart + v).astype("datetime64[M]").astype("datetime64[ms]").astype(np.int64)
    return start, end


def _infer_base_ms(t: np.ndarray) -> int:
    if len(t) < 2:
        raise ValueError("cannot infer base timeframe from < 2 bars; pass base_tf")
    return int(np.median(np.diff(t)))


def request_security_htf(df: pd.DataFrame, tf, func: Callable, base_tf=None):
    """``request.security(syminfo.tickerid, tf, expr, lookahead=barmerge.lookahead_off)``
    for historical bars, without repainting.

    * ``df``: base-timeframe OHLCV (DatetimeIndex or 'open_time' column; columns
      open/high/low/close[/volume/...]).
    * Base bars are aggregated into ``tf`` buckets (UTC-aligned: minutes/days from the
      epoch, weeks from Monday 00:00, calendar months). Only HTF bars that are *closed*
      (bucket end <= close time of the last base bar) are passed to ``func``.
    * ``func(htf_df)`` returns an array, a tuple of arrays, or a dict of arrays, one value
      per HTF bar.
    * Each base bar ``i`` receives the value of the latest HTF bar whose end time is
      <= the close time of bar ``i``: the HTF value becomes visible on the base bar that
      completes the HTF bar (TradingView's historical ``lookahead_off`` behaviour) and is
      forward-filled until the next HTF bar closes. Before the first closed HTF bar: NaN
      (False for bool results).
    * If ``tf`` is not larger than the base timeframe the function is evaluated on the
      base bars directly (Pine returns the chart-TF / last-intrabar value; requesting a
      lower timeframe than the chart cannot be reproduced from base data).
    """
    t = times_ms(df)
    base_ms = tf_ms(base_tf) if base_tf is not None else _infer_base_ms(t)
    kind, v = parse_tf(tf)
    out_cols = [c for c in df.columns if c not in ("open_time", "time", "timestamp")]

    if tf_ms(tf) <= base_ms:
        res = func(df)
        return res

    start, end = _bucket_start(t, kind, v)
    last_close = t[-1] + base_ms
    # group boundaries (data sorted by time)
    newgrp = np.ones(len(t), dtype=bool)
    newgrp[1:] = start[1:] != start[:-1]
    first_idx = np.flatnonzero(newgrp)
    g_start = start[first_idx]
    g_end = end[first_idx]
    closed = g_end <= last_close
    last_idx = np.append(first_idx[1:] - 1, len(t) - 1)

    agg = {}
    for c in out_cols:
        col = df[c].to_numpy()
        if not np.issubdtype(col.dtype, np.number):
            continue
        col = col.astype(np.float64)
        lc = c.lower()
        if lc == "open":
            agg[c] = col[first_idx]
        elif lc == "close":
            agg[c] = col[last_idx]
        elif lc == "high":
            agg[c] = np.maximum.reduceat(col, first_idx)
        elif lc == "low":
            agg[c] = np.minimum.reduceat(col, first_idx)
        else:
            agg[c] = np.add.reduceat(col, first_idx)
    htf = pd.DataFrame(agg, index=pd.to_datetime(g_start, unit="ms", utc=True))
    htf.index.name = "open_time"
    htf = htf.iloc[np.flatnonzero(closed)]
    htf_end = g_end[closed]

    base_close = t + base_ms
    j = np.searchsorted(htf_end, base_close, side="right") - 1
    valid = j >= 0
    jj = np.where(valid, j, 0)

    def _map(a):
        a = np.asarray(a)
        if a.dtype == bool:
            out = np.zeros(len(t), dtype=bool)
            if len(a):
                out[valid] = a[jj[valid]]
            return out
        a = a.astype(np.float64)
        out = np.full(len(t), np.nan)
        if len(a):
            out[valid] = a[jj[valid]]
        return out

    if len(htf) == 0:
        res = func(df.iloc[:0])
    else:
        res = func(htf)
    if isinstance(res, dict):
        return {k: _map(a) for k, a in res.items()}
    if isinstance(res, tuple):
        return tuple(_map(a) for a in res)
    return _map(res)
