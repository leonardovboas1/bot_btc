"""Hand-computed values for the Pine ta.* primitives."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from btclab.indicators import base as ta

nan = np.nan


def test_sma():
    assert_allclose(ta.sma([1, 2, 3, 4, 5], 3), [nan, nan, 2, 3, 4])
    # na inside the window propagates (Pine sum over x[i])
    assert_allclose(ta.sma([1, nan, 3, 4, 5, 6], 2), [nan, nan, nan, 3.5, 4.5, 5.5])


def test_ema_sma_seed():
    # alpha = 2/(3+1) = .5 ; seed = sma(1,2,3) = 2
    assert_allclose(ta.ema([1, 2, 3, 4, 5], 3), [nan, nan, 2.0, 3.0, 4.0])
    assert_allclose(ta.ema([2, 4, 6, 10], 3), [nan, nan, 4.0, 7.0])


def test_ema_leading_nan_reseed():
    # leading na: seed is the SMA of the first 3 valid values
    assert_allclose(ta.ema([nan, nan, 1, 2, 3, 4], 3), [nan, nan, nan, nan, 2.0, 3.0])


def test_rma():
    # alpha = 1/3, seed 2 ; 4/3 + 2*2/3 = 8/3 ; 5/3 + 2/3*8/3 = 31/9
    assert_allclose(ta.rma([1, 2, 3, 4, 5], 3), [nan, nan, 2.0, 8 / 3, 31 / 9])


def test_rsi_hand():
    # x = 1 2 3 2 4 ; u = . 1 1 0 2 ; d = . 0 0 1 0 ; rma(2): up 1, .5, 1.25 ; dn 0, .5, .25
    r = ta.rsi([1, 2, 3, 2, 4], 2)
    assert_allclose(r, [nan, nan, 100.0, 50.0, 100 - 100 / 6])


def test_rsi_zero_down_and_zero_up():
    assert_allclose(ta.rsi(np.arange(10.0), 3)[3:], 100.0)        # down == 0 -> 100
    assert_allclose(ta.rsi(-np.arange(10.0), 3)[3:], 0.0)         # up == 0   -> 0
    assert_allclose(ta.rsi(np.ones(10), 3)[3:], 100.0)            # both 0: Pine v4 rule down==0 first


def test_wma_hma():
    assert_allclose(ta.wma([1, 2, 3, 4], 3), [nan, nan, (1 + 4 + 9) / 6, (2 + 6 + 12) / 6])
    x = np.arange(1.0, 30.0)
    # linear input: wma lags by (n-1)/3 ; HMA of a line equals the line (after warm-up)
    h = ta.hma(x, 9)
    assert_allclose(h[~np.isnan(h)], x[~np.isnan(h)], atol=1e-9)


def test_tr_atr():
    h, l, c = [10, 12, 11], [8, 9, 7], [9, 11, 8]
    assert_allclose(ta.tr(h, l, c, True), [2, 3, 4])
    assert_allclose(ta.tr(h, l, c, False), [nan, 3, 4])
    assert_allclose(ta.atr(h, l, c, 2), [nan, 2.5, 3.25])


def test_change_highest_lowest_stoch():
    x = [3, 1, 4, 1, 5, 9, 2]
    assert_allclose(ta.change(x), [nan, -2, 3, -3, 4, 4, -7])
    assert_allclose(ta.change(x, 2), [nan, nan, 1, 0, 1, 8, -3])
    assert_allclose(ta.highest(x, 3), [nan, nan, 4, 4, 5, 9, 9])
    assert_allclose(ta.lowest(x, 3), [nan, nan, 1, 1, 1, 1, 2])
    s = ta.stoch(x, x, x, 3)
    assert_allclose(s, [nan, nan, 100, 0, 100, 100, 0])
    assert np.isnan(ta.stoch([1, 1, 1], [1, 1, 1], [1, 1, 1], 3)[2])  # 0/0 -> na


def test_crosses():
    a = [1, 2, 3, 2, 1]
    assert_array_equal(ta.crossover(a, 2), [False, False, True, False, False])
    assert_array_equal(ta.crossunder(a, 2), [False, False, False, False, True])
    assert_array_equal(ta.cross(a, 2), [False, False, True, False, True])
    assert ta.crossover([nan, 3], [1, 1]).tolist() == [False, False]


def test_valuewhen():
    cond = [False, True, False, True, False]
    src = [10, 11, 12, 13, 14]
    assert_allclose(ta.valuewhen(cond, src, 0), [nan, 11, 11, 13, 13])
    assert_allclose(ta.valuewhen(cond, src, 1), [nan, nan, nan, 11, 11])
    # float condition: na and 0 are false (Pine coercion)
    assert_allclose(ta.valuewhen([nan, 0.0, 2.0, nan], src[:4], 0), [nan, nan, 12, 12])


def test_pivots_at_confirmation_bar():
    x = [1, 3, 2, 5, 4, 3, 6]
    ph = ta.pivothigh(x, 1, 1)
    assert_allclose(ph, [nan, nan, 3, nan, 5, nan, nan])
    pl = ta.pivotlow(x, 1, 1)
    assert_allclose(pl, [nan, nan, nan, 2, nan, nan, 3])
    ph2 = ta.pivothigh(x, 2, 2)
    assert_allclose(ph2, [nan, nan, nan, nan, nan, 5, nan])


def test_nz_sources_heikinashi():
    assert_allclose(ta.nz([nan, 1.0]), [0.0, 1.0])
    assert_allclose(ta.nz([nan], 5), [5.0])
    assert_allclose(ta.hl2([4], [2]), [3])
    assert_allclose(ta.hlc3([4], [2], [3]), [3])
    ho, hh, hl, hc = ta.heikinashi([10, 11], [12, 13], [9, 10], [11, 12])
    assert_allclose(hc, [10.5, 11.5])
    assert_allclose(ho, [10.5, 10.5])
    assert_allclose(hh, [12, 13])
    assert_allclose(hl, [9, 10])


def test_parse_tf():
    assert ta.parse_tf("240") == ("ms", 240 * 60_000)
    assert ta.parse_tf("1D") == ("ms", 86_400_000)
    assert ta.parse_tf("D") == ("ms", 86_400_000)
    assert ta.parse_tf("4h") == ("ms", 4 * 3_600_000)
    assert ta.parse_tf("W") == ("W", 1)
    assert ta.parse_tf("M") == ("M", 1)
    assert ta.parse_tf("15m") == ("ms", 15 * 60_000)
