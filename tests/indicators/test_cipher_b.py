import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from btclab.indicators import base as ta
from btclab.indicators import cipher_b as cb

from conftest import synthetic_ohlcv

#                0  1    2    3    4    5    6    7    8    9   10   11   12   13   14
SRC = np.array([0, 0, -70, -80, -90, -80, -70, -60, -50, -60, -70, -85, -75, -65, -60], float)
LOW = np.full(15, 110.0)
LOW[4], LOW[11] = 100.0, 95.0
HIGH = LOW + 5


def test_find_divs_regular_bull_at_confirmation_bar():
    d = cb.find_divs(SRC, HIGH, LOW, 0, 0, False)
    assert np.flatnonzero(~np.isnan(d["fractal_bot"])).tolist() == [6, 13]
    assert d["bull"].tolist() == [i == 13 for i in range(15)]
    assert d["low_prev"][13] == -90
    assert not d["bull_hidden"].any()
    # limits: both bottoms <= -65 -> still a divergence
    assert cb.find_divs(SRC, HIGH, LOW, 45, -65, True)["bull"][13]
    # a stricter limit excludes the second fractal
    assert not cb.find_divs(SRC, HIGH, LOW, 45, -88, True)["bull"].any()


def test_find_divs_hidden_bull():
    low = LOW.copy()
    low[11] = 105.0           # price higher low
    src = SRC.copy()
    src[11] = -95.0            # indicator lower low
    src[10], src[12] = -80, -85
    d = cb.find_divs(src, HIGH, low, 0, 0, False)
    assert d["bull_hidden"][13] and not d["bull"][13]


def test_fractal_zero_value_is_false():
    """Pine coerces a float fractal value of exactly 0 to false."""
    src = np.array([5, 4, 0, 4, 5, 6], float)
    d = cb.find_divs(src, src, src, 0, 0, False)
    assert d["fractal_bot"][4] == 0.0
    assert np.isnan(ta.valuewhen(d["fractal_bot"], src, 0)[4])


def test_buy_sell_definition(ohlcv):
    o = cb.cipher_b(*(ohlcv[k].to_numpy() for k in ("open", "high", "low", "close")), time=ohlcv.index)
    wt1, wt2 = o["wt1"], o["wt2"]
    cross_up = ta.crossover(wt1, wt2)
    assert np.array_equal(o["buy"], cross_up & (wt2 <= -53))
    assert np.array_equal(o["sell"], ta.crossunder(wt1, wt2) & (wt2 >= 53))
    assert o["buy"].sum() > 0 and o["sell"].sum() > 0
    for k, v in o.items():
        assert len(v) == len(ohlcv), k


def test_rsi_mfi_and_stc_ranges(ohlcv):
    o = cb.cipher_b(*(ohlcv[k].to_numpy() for k in ("open", "high", "low", "close")), time=ohlcv.index)
    m = o["rsi_mfi"][~np.isnan(o["rsi_mfi"])]
    assert np.all(np.abs(m + 2.5) <= 150 + 1e-9)
    s = o["stc"][100:]
    assert np.all((s >= -1e-9) & (s <= 100 + 1e-9))
    k = o["stoch_k"][~np.isnan(o["stoch_k"])]
    assert np.all((k >= 0) & (k <= 100))


def test_stc_reference():
    """Line-by-line Python transcription of f_tc."""
    c = synthetic_ohlcv(600)["close"].to_numpy()
    m = ta.ema(c, 23) - ta.ema(c, 50)
    lo, hi = ta.lowest(m, 10), ta.highest(m, 10)
    g, d, e, s = [], [], [], []
    for i in range(len(c)):
        beta = hi[i] - lo[i]
        gi = (m[i] - lo[i]) / beta * 100 if beta > 0 else (g[-1] if g and not np.isnan(g[-1]) else 0.0)
        g.append(gi)
        di = gi if (not d or np.isnan(d[-1])) else d[-1] + 0.5 * (gi - d[-1])
        d.append(di)
        win = d[-10:]
        if len(win) < 10:
            eps = zeta = np.nan
        else:
            eps = min(win); zeta = max(win) - eps
        ei = (di - eps) / zeta * 100 if zeta > 0 else (e[-1] if e and not np.isnan(e[-1]) else 0.0)
        e.append(ei)
        si = ei if (not s or np.isnan(s[-1])) else s[-1] + 0.5 * (ei - s[-1])
        s.append(si)
    assert_allclose(cb.schaff_tc(c), s, rtol=1e-10, atol=1e-10)


def test_sommi_flag_uses_closed_htf_only():
    df = synthetic_ohlcv(1200, freq="4h")
    args = [df[k].to_numpy() for k in ("open", "high", "low", "close")]
    o = cb.cipher_b(*args, time=df.index)
    hv = o["htf_wt_vwap"]
    # 12h bars close on the 4h bars starting 08:00 and 20:00 -> value only changes there
    changed = np.flatnonzero(np.diff(hv) != 0) + 1
    changed = changed[~np.isnan(hv[changed])]
    assert set(df.index[changed].hour) <= {8, 20}
    # without timestamps the HTF parts are disabled, not wrong
    o2 = cb.cipher_b(*args)
    assert not o2["sommi_bull_flag"].any() and np.isnan(o2["htf_wt_vwap"]).all()


def test_unknown_param_raises():
    with pytest.raises(TypeError):
        cb.cipher_b([1.0], [1.0], [1.0], [1.0], bogus=1)
