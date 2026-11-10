"""Regressions from the line-by-line diff against the user's verbatim Pine sources
(research/pine/vmc_cipher_b_divergences.pine, research/pine/brayvb_strategy_v2.pine)."""
import numpy as np
import pytest

from btclab.indicators import base as ta
from btclab.indicators import cipher_b as cb
from btclab.indicators import gaussian_channel as gcm
from btclab.indicators import stoch_rsi as sr
from btclab.indicators.registry import REGISTRY, STRATEGIES, compute


def _cb(df, **p):
    return cb.cipher_b(*(df[k].to_numpy() for k in ("open", "high", "low", "close")), time=df.index, **p)


def test_buy_div_not_gated_by_addshow(ohlcv):
    """buySignalDiv = (wtShowDiv and wtBullDiv) or (wtShowDiv and wtBullDiv_add) or ...
    wtDivOBLevel_addshow is colour-only and must not exist as a signal parameter."""
    o = _cb(ohlcv)
    assert np.array_equal(o["bull_div"], o["wt_bull_div"] | o["wt_bull_div_add"])
    assert np.array_equal(o["bear_div"], o["wt_bear_div"] | o["wt_bear_div_add"])
    with pytest.raises(TypeError):
        _cb(ohlcv.iloc[:50], wt_div_add_show=False)
    off = _cb(ohlcv, wt_show_div=False)
    assert not off["bull_div"].any() and not off["gold_buy"].any()
    on = _cb(ohlcv, stoch_show_div=True, rsi_show_div=True)
    assert np.array_equal(on["bull_div"], o["bull_div"] | o["stoch_bull_div"] | o["rsi_bull_div"])


def test_wt_cross_up_down_are_states(ohlcv):
    """wtCrossUp = wt2 - wt1 <= 0 ; wtCrossDown = wt2 - wt1 >= 0 (states, not events)."""
    o = _cb(ohlcv)
    diff = o["wt2"] - o["wt1"]
    assert np.array_equal(o["wt_up_state"], diff <= 0)
    assert np.array_equal(o["wt_down_state"], diff >= 0)
    assert np.array_equal(o["wt_cross_up"], o["wt_cross"] & (diff <= 0))
    assert np.array_equal(o["buy"], o["wt_cross"] & (diff <= 0) & (o["wt2"] <= -53))


def test_stochrsi_argument_order_and_log(ohlcv):
    """f_stochrsi(stochSRC, stochLen, stochRsiLen, ...): rsi(log(close), stochRsiLen),
    stoch(rsi, rsi, rsi, stochLen); k = kk unless stochAvg, d = sma(kk)."""
    c = ohlcv["close"].to_numpy()
    o = _cb(ohlcv, stoch_len=10, stoch_rsi_len=20)
    r = ta.rsi(np.log(c), 20)
    kk = ta.sma(ta.stoch(r, r, r, 10), 3)
    d1 = ta.sma(kk, 3)
    assert np.array_equal(o["stoch_k"], kk, equal_nan=True)
    assert np.array_equal(o["stoch_d"], d1, equal_nan=True)
    oa = _cb(ohlcv, stoch_avg=True)
    k0, d0 = sr.stoch_rsi_kd(c, 14, 14, 3, 3, True)
    assert np.array_equal(oa["stoch_k"], (k0 + d0) / 2, equal_nan=True)
    assert np.array_equal(oa["stoch_d"], d0, equal_nan=True)


def test_rsimfi_formula(ohlcv):
    o_, h, l, c = (ohlcv[k].to_numpy() for k in ("open", "high", "low", "close"))
    ref = ta.sma((c - o_) / (h - l) * 150.0, 60) - 2.5
    assert np.allclose(_cb(ohlcv)["rsi_mfi"], ref, equal_nan=True, rtol=0, atol=1e-12)


def test_gold_buy_conditions(ohlcv):
    o = _cb(ohlcv)
    r = o["rsi"]
    d = cb.find_divs(o["wt2"], ohlcv["high"].to_numpy(), ohlcv["low"].to_numpy(), 45, -65, True)
    last_rsi = ta.shift(ta.valuewhen(d["fractal_bot"], ta.shift(r, 2), 0), 2)
    lp = d["low_prev"]
    ref = d["bull"] & (lp <= -75) & (o["wt2"] > -75) & ((lp - o["wt2"]) <= -5) & (last_rsi < 30)
    assert np.array_equal(o["gold_buy"], ref)


def test_find_divs_offsets():
    """highPrev = valuewhen(fractalTop, src[2], 0)[2] — the [2] after valuewhen means the
    current fractal is excluded, so a divergence compares against the PREVIOUS fractal."""
    src = np.array([0, 1, 5, 1, 0, 1, 4, 1, 0], float)   # tops at 2 (5) and 6 (4)
    high = np.array([0, 0, 10, 0, 0, 0, 12, 0, 0], float)
    d = cb.find_divs(src, high, high, 0, 0, False)
    assert d["bear"].tolist() == [i == 8 for i in range(9)]  # price HH, osc LH


def test_brayvb_v2_spec(ohlcv):
    assert "brayvb_v2" in REGISTRY and "brayvb_v2" in STRATEGIES
    out = compute("brayvb_v2", ohlcv)
    h, l, c = (ohlcv[k].to_numpy() for k in ("high", "low", "close"))
    g = gcm.gaussian_channel(h, l, c)
    k, _ = sr.stoch_rsi_kd(c, 14, 14, 3, 3, False)
    entry = (g["filt"] > ta.shift(g["filt"])) & (c > g["hband"]) & ((k >= 80) | (k <= 20))
    assert np.array_equal(out["long_entry"], entry)
    assert np.array_equal(out["long_exit"], ta.crossunder(c, g["hband"]))
    assert out["long_entry"].any() and out["long_exit"].any()
    # k == 80 exactly counts as extreme (>=), unlike k_above_ob (>)
    s = sr.stoch_rsi(np.r_[np.linspace(1, 50, 60), 50.0])
    assert np.array_equal(s["k_extreme"], (s["k"] >= 80) | (s["k"] <= 20))
