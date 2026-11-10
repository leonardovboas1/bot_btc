import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from btclab.indicators import base as ta
from btclab.indicators import gaussian_channel as gcm

from conftest import load_real


def cascade(alpha, src, n):
    """N cascaded zero-initialised one-pole filters y = a*x + (1-a)*y[1]."""
    y = np.nan_to_num(np.asarray(src, float))
    for _ in range(n):
        out = np.zeros_like(y)
        prev = 0.0
        for i, v in enumerate(y):
            prev = alpha * v + (1 - alpha) * prev
            out[i] = prev
        y = out
    return y


def test_alpha_formula():
    per, N = 144, 4
    beta = (1 - math.cos(4 * math.asin(1) / per)) / (math.pow(1.414, 2 / N) - 1)
    assert gcm.gc_alpha(per, N) == pytest.approx(-beta + math.sqrt(beta ** 2 + 2 * beta))


@pytest.mark.parametrize("poles", range(1, 10))
def test_npole_equals_cascade(poles, synth):
    """f_filt9x's binomial recursion is algebraically N cascaded EMAs."""
    src = synth["close"].to_numpy() / 30000.0
    a = gcm.gc_alpha(144, poles)
    # the expanded binomial recursion amplifies rounding error with the pole count
    # (measured ~6e-7 relative at 9 poles); the cascade is the better-conditioned form
    rtol = 1e-9 if poles <= 4 else 5e-6
    assert_allclose(gcm.f_filt9x(a, src, poles), cascade(a, src, poles), rtol=rtol, atol=1e-9)


def test_one_pole_first_values():
    a = 0.25
    f = gcm.f_filt9x(a, [4.0, 8.0, np.nan, 4.0], 1)
    # nz(src) and nz(f[1]) -> zero initialisation, na source treated as 0
    assert_allclose(f, [1.0, 1.0 * 0.75 + 2.0, (2.75) * 0.75, 2.0625 * 0.75 + 1.0])


def test_f_pole_quirk_and_selection(synth):
    src = synth["close"].to_numpy()
    a = gcm.gc_alpha(100, 7)
    for n in (1, 3, 7, 9):
        fn, f1 = gcm.f_pole(a, src, n)
        assert_allclose(fn, gcm.f_filt9x(a, src, n))
        assert_allclose(f1, gcm.f_filt9x(a, src, 1))
    with pytest.raises(ValueError):
        gcm.f_pole(a, src, 10)


def test_reduced_lag_uses_truncated_lag(synth):
    h, l, c = (synth[k].to_numpy() for k in ("high", "low", "close"))
    out = gcm.gaussian_channel(h, l, c, poles=4, period=144, mode_lag=True)
    s = ta.hlc3(h, l, c)
    lag = int(143 / 8)
    assert lag == 17
    srcdata = s + (s - ta.shift(s, lag))
    a = gcm.gc_alpha(144, 4)
    assert_allclose(out["filt"], gcm.f_filt9x(a, srcdata, 4))


def test_fast_mode(synth):
    h, l, c = (synth[k].to_numpy() for k in ("high", "low", "close"))
    a = gcm.gc_alpha(144, 4)
    s = ta.hlc3(h, l, c)
    out = gcm.gaussian_channel(h, l, c, mode_fast=True)
    assert_allclose(out["filt"], (gcm.f_filt9x(a, s, 4) + gcm.f_filt9x(a, s, 1)) / 2)


@pytest.mark.parametrize("tf", ["1d", "4h"])
def test_smoke_real_data(tf):
    df = load_real(tf)
    if df is None:
        pytest.skip("real data not downloaded")
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    for kw in ({}, {"mode_lag": True}, {"mode_fast": True}, {"poles": 9, "period": 60}, {"poles": 1}):
        out = gcm.gaussian_channel(h, l, c, **kw)
        w = gcm.warmup_bars(**{"poles": 4, "period": 144, **kw})
        assert w < len(c) // 2
        f, hb, lb = out["filt"][w:], out["hband"][w:], out["lband"][w:]
        assert np.isfinite(f).all() and np.isfinite(hb).all()
        assert (hb > f).all() and (f > lb).all()
        # the filter tracks price: median relative distance well below 25 %
        assert np.median(np.abs(f / c[w:] - 1)) < 0.25
        for k in ("filter_rising", "close_above_hband", "cross_below_hband"):
            assert out[k].dtype == bool and 0 < out[k][w:].sum() < len(f)
    # default daily channel: rising filter + close above upper band happens in bull runs
    out = gcm.gaussian_channel(h, l, c)
    assert (out["filter_rising"] & out["close_above_hband"]).sum() > 0


def test_warmup_converges(synth):
    h, l, c = (synth[k].to_numpy() for k in ("high", "low", "close"))
    w = gcm.warmup_bars(poles=4, period=144)
    full = gcm.gaussian_channel(h, l, c)["filt"]
    later = gcm.gaussian_channel(h[300:], l[300:], c[300:])["filt"]
    # starting 300 bars later: identical (to tolerance) once both have warmed up
    assert_allclose(later[w:], full[300 + w:], rtol=1e-6)
