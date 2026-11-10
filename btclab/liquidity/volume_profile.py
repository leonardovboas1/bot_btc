"""Rolling volume-at-price from klines: bar volume spread uniformly over its high-low range.

Bins are absolute log-price bins (shift-invariant), so values at t depend on bars <= t only.
"""
from __future__ import annotations

import numpy as np

from ._util import LOG_STEP, bin_center, bin_index


def _bars_to_bins(high, low, volume, step):
    ih, il = bin_index(high, step), bin_index(low, step)
    return il, ih, np.asarray(volume, float)


def profile_stats(prof: np.ndarray, value_area: float = 0.70, smooth: int = 2,
                  hvn_q: float = 0.8, lvn_q: float = 0.2):
    """POC / VAH / VAL bin offsets and HVN/LVN bin lists for one profile vector."""
    tot = prof.sum()
    if tot <= 0:
        return None
    poc = int(np.argmax(prof))
    lo = hi = poc
    acc = prof[poc]
    while acc < value_area * tot and (lo > 0 or hi < len(prof) - 1):
        up = prof[hi + 1] if hi + 1 < len(prof) else -1
        dn = prof[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1; acc += up
        else:
            lo -= 1; acc += dn
    k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    sm = np.convolve(prof, k, mode="same")
    nz = sm[sm > 0]
    peaks = (sm[1:-1] > sm[:-2]) & (sm[1:-1] >= sm[2:])
    troughs = (sm[1:-1] < sm[:-2]) & (sm[1:-1] <= sm[2:])
    idx = np.arange(1, len(sm) - 1)
    hq, lq = np.quantile(nz, hvn_q), np.quantile(nz, lvn_q)
    hvn = idx[peaks & (sm[1:-1] >= hq)]
    lvn = idx[troughs & (sm[1:-1] <= lq) & (sm[1:-1] > 0)]
    return poc, hi, lo, hvn, lvn


def rolling_profile(high, low, volume, window: int = 168, step: float = LOG_STEP,
                    value_area: float = 0.70, nodes: bool = False):
    """Per-bar rolling POC/VAH/VAL price arrays (NaN until `window` bars exist).

    With nodes=True also returns lists of HVN / LVN prices per bar."""
    high, low, volume = (np.asarray(a, float) for a in (high, low, volume))
    T = len(high)
    il, ih, vol = _bars_to_bins(high, low, volume, step)
    base = int(il.min())
    P = int(ih.max()) - base + 1
    prof = np.zeros(P)
    poc = np.full(T, np.nan); vah = poc.copy(); val = poc.copy()
    hvn = [[] for _ in range(T)]; lvn = [[] for _ in range(T)]

    def add(t, sign):
        a, b = il[t] - base, ih[t] - base
        prof[a:b + 1] += sign * vol[t] / (b - a + 1)

    for t in range(T):
        add(t, 1)
        if t >= window:
            add(t - window, -1)
        if t >= window - 1:
            np.maximum(prof, 0, out=prof)
            r = profile_stats(prof, value_area)
            if r is None:
                continue
            p, h, l, hv, lv = r
            c = lambda i: bin_center(i + base, step)
            poc[t], vah[t], val[t] = c(p), c(h), c(l)
            if nodes:
                hvn[t] = [float(x) for x in c(hv)]
                lvn[t] = [float(x) for x in c(lv)]
    out = {"poc": poc, "vah": vah, "val": val}
    if nodes:
        out.update(hvn=hvn, lvn=lvn)
    return out


def profile_snapshot(high, low, volume, step: float = LOG_STEP, value_area: float = 0.70) -> dict:
    """Profile over ALL given bars (caller slices the window); JSON-serializable dict."""
    il, ih, vol = _bars_to_bins(high, low, volume, step)
    base = int(il.min()); P = int(ih.max()) - base + 1
    prof = np.zeros(P)
    for a, b, v in zip(il - base, ih - base, vol):
        prof[a:b + 1] += v / (b - a + 1)
    prices = bin_center(np.arange(base, base + P), step)
    r = profile_stats(prof, value_area)
    d = {"prices": prices.tolist(), "volume": prof.tolist()}
    if r:
        p, h, l, hv, lv = r
        d.update(poc=float(prices[p]), vah=float(prices[h]), val=float(prices[l]),
                 hvn=[float(prices[i]) for i in hv], lvn=[float(prices[i]) for i in lv])
    return d
