"""Gaussian Channel [DW] (DonovanWall, Pine v4) — Pine-faithful port.

Original logic (abridged)::

    beta  = (1 - cos(4*asin(1)/per)) / (pow(1.414, 2/N) - 1)
    alpha = -beta + sqrt(pow(beta, 2) + 2*beta)
    lag   = (per - 1)/(2*N)
    srcdata = modeLag ? src + (src - src[lag]) : src
    trdata  = modeLag ? tr(true) + (tr(true) - tr(true)[lag]) : tr(true)
    [filtn, filt1]     = f_pole(alpha, srcdata, N)
    [filtntr, filt1tr] = f_pole(alpha, trdata,  N)
    filt   = modeFast ? (filtn + filt1)/2 : filtn
    filttr = modeFast ? (filtntr + filt1tr)/2 : filtntr
    hband = filt + filttr*mult ; lband = filt - filttr*mult

``f_filt9x`` is the N-pole recursion
``f = a^N*nz(s) + sum_k (-1)^(k+1) C(N,k) (1-a)^k nz(f[k])``, i.e. N cascaded
one-pole EMAs with the same alpha, zero-initialised (Pine ``nz``). There is
therefore no NaN warm-up: the filter starts at 0 and converges; use
``warmup_bars`` from the registry before trusting it.

Fidelity notes
* ``1.414`` (not sqrt(2)) is used in beta, as in the original.
* ``src[lag]`` with a fractional ``lag``: Pine truncates the history index;
  we use ``int(lag)`` (17 for the 144/4 defaults).
* During reduced-lag warm-up ``src[lag]`` is na, so ``nz`` feeds 0 into the
  filter exactly as Pine does.
* The ``_f7 = (_i >= 2 ? f_filt9x(_a, _s, 7) : 0)`` quirk in ``f_pole`` is
  replicated (the 7-pole filter is computed whenever N >= 2). Because ``_i`` is a
  constant it does not change the output (``_fn`` only selects ``_f7`` when N == 7).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .base import crossover, crossunder, f64, hlc3, shift, tr

# binomial weights _m2.._m9 exactly as in the Pine source (index = pole count)
_W = np.zeros((10, 10))
for _i in range(1, 10):
    for _k in range(0, _i + 1):
        _W[_i, _k] = math.comb(_i, _k)


@njit(cache=True)
def _filt9x(a, s, i, W):
    n = len(s)
    f = np.zeros(n)
    x = 1.0 - a
    ai = a ** i
    for t in range(n):
        sv = s[t]
        if np.isnan(sv):
            sv = 0.0
        acc = ai * sv
        # + i*x*f[1]  - m2*x^2*f[2] + m3*x^3*f[3] - ...  (left-to-right like Pine)
        for k in range(1, 10):
            if k > i:
                break
            prev = f[t - k] if t - k >= 0 else 0.0
            term = (W[i, k] * x ** k) * prev
            if k % 2 == 1:
                acc = acc + term
            else:
                acc = acc - term
        f[t] = acc
    return f


def f_filt9x(alpha: float, src, poles: int) -> np.ndarray:
    return _filt9x(float(alpha), f64(src), int(poles), _W)


def f_pole(alpha: float, src, poles: int):
    """Returns (filt_n, filt_1). Replicates the original's evaluation pattern."""
    i = int(poles)
    if not 1 <= i <= 9:
        raise ValueError("poles must be in 1..9")
    f = {1: f_filt9x(alpha, src, 1)}
    for k in range(2, 10):
        # original: _f7 = (_i >= 2 ? f_filt9x(_a, _s, 7) : 0)   <- quirk kept
        cond = (i >= 2) if k == 7 else ((i == 9) if k == 9 else (i >= k))
        if cond:
            f[k] = f_filt9x(alpha, src, k)
    return f[i], f[1]


def gc_alpha(period: float, poles: int) -> float:
    beta = (1.0 - math.cos(4.0 * math.asin(1.0) / period)) / (math.pow(1.414, 2.0 / poles) - 1.0)
    return -beta + math.sqrt(beta * beta + 2.0 * beta)


def gaussian_channel(high, low, close, src=None, poles: int = 4, period: int = 144,
                     mult: float = 1.414, mode_lag: bool = False, mode_fast: bool = False) -> dict:
    """Gaussian Channel. ``src`` defaults to hlc3. Returns filt/hband/lband/filt_tr and
    bool primitives evaluated on ``close`` (the usual strategy usage)."""
    h, l, c = f64(high), f64(low), f64(close)
    s = hlc3(h, l, c) if src is None else f64(src)
    N = int(poles)
    alpha = gc_alpha(float(period), N)
    lag = int((period - 1) / (2 * N))
    trv = tr(h, l, c, True)
    if mode_lag:
        srcdata = s + (s - shift(s, lag))
        trdata = trv + (trv - shift(trv, lag))
    else:
        srcdata, trdata = s, trv
    filtn, filt1 = f_pole(alpha, srcdata, N)
    filtntr, filt1tr = f_pole(alpha, trdata, N)
    filt = (filtn + filt1) / 2.0 if mode_fast else filtn
    filttr = (filtntr + filt1tr) / 2.0 if mode_fast else filtntr
    hband = filt + filttr * mult
    lband = filt - filttr * mult
    f1 = shift(filt)
    return {
        "filt": filt,
        "hband": hband,
        "lband": lband,
        "filt_tr": filttr,
        "filter_rising": filt > f1,
        "filter_falling": filt < f1,
        "close_above_hband": c > hband,
        "close_below_lband": c < lband,
        "close_above_filt": c > filt,
        "close_below_filt": c < filt,
        "cross_above_hband": crossover(c, hband),
        "cross_below_hband": crossunder(c, hband),
        "cross_above_filt": crossover(c, filt),
        "cross_below_filt": crossunder(c, filt),
        "cross_below_lband": crossunder(c, lband),
        "cross_above_lband": crossover(c, lband),
    }


def warmup_bars(poles: int = 4, period: int = 144, mode_lag: bool = False, tol: float = 1e-8, **_) -> int:
    """Bars until the zero-initialised N-pole filter has forgotten its initial state:
    smallest t with C(t+N-1, N-1) * (1-alpha)^t < tol (impulse-response tail bound)."""
    N = int(poles)
    x = 1.0 - gc_alpha(float(period), N)
    t = 1
    while math.comb(t + N - 1, N - 1) * x ** t >= tol and t < 100_000:
        t += 1
    return t + (int((period - 1) / (2 * N)) if mode_lag else 0)
