"""Catalog-2 indicators: known-value hand checks + causality of every primitive."""
import numpy as np
import pytest

from btclab.indicators import cycle as cy
from btclab.indicators import structure as st
from btclab.indicators import trend2 as t2
from btclab.indicators import volatility as vo
from btclab.indicators import volume as vl
from btclab.indicators.registry import REGISTRY, compute

NEW = ["bb", "kc", "squeeze", "donchian", "chandelier", "ichimoku", "hull", "ssl", "utbot", "rangefilter",
       "qqe", "obv", "cmf", "vwap", "avwap", "cvd", "swing", "structure", "fvg", "ob", "mayer", "picycle"]
CUTS = (300, 777, 1200)


# ------------------------------------------------------------------ causality (all primitives)
@pytest.mark.parametrize("name", NEW)
def test_primitives_causal_on_append(name, synth):
    df = synth.iloc[:1400]
    full = compute(name, df)
    for prim in REGISTRY[name].primitives:
        key = prim.split(".", 1)[1]
        for k in CUTS:
            part = compute(name, df.iloc[:k])[key]
            assert np.array_equal(part, full[key][:k]), (prim, k)


@pytest.mark.parametrize("name", NEW)
def test_primitives_causal_future_scramble(name, synth):
    df = synth.iloc[:1400]
    k = 900
    full = compute(name, df)
    pert = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        pert.iloc[k:, pert.columns.get_loc(c)] = df[c].to_numpy()[k:][::-1]
    other = compute(name, pert)
    for prim in REGISTRY[name].primitives:
        key = prim.split(".", 1)[1]
        assert np.array_equal(other[key][:k], full[key][:k]), prim


@pytest.mark.parametrize("name", NEW)
def test_registered_bool_and_fires(name, synth):
    out = compute(name, synth)
    for prim in REGISTRY[name].primitives:
        assert out[prim.split(".", 1)[1]].dtype == bool


# ------------------------------------------------------------------ known values
def test_bollinger_hand():
    c = np.array([1.0, 2, 3, 4, 5])
    o = vo.bollinger(c, 3, 2.0)
    assert np.isnan(o["basis"][1])
    assert o["basis"][2] == pytest.approx(2.0)
    sd = np.sqrt(2 / 3)  # population stdev of 1,2,3
    assert o["upper"][2] == pytest.approx(2 + 2 * sd)
    assert o["lower"][2] == pytest.approx(2 - 2 * sd)


def test_linreg_exact_line():
    y = 3.0 + 2.0 * np.arange(10)
    assert np.allclose(vo.linreg(y, 5)[4:], y[4:])


def test_donchian_breakout():
    h = np.array([5, 6, 7, 6, 6, 9.0])
    c = np.array([5, 6, 6, 6, 6, 9.0])
    o = vo.donchian(h, h - 1, c, 3)
    assert o["breakout_up"].tolist() == [False] * 5 + [True]


def test_squeeze_fire_on_release():
    rng = np.random.default_rng(0)
    c = 100 + np.concatenate([rng.normal(0, 0.05, 80), np.cumsum(rng.normal(0, 3, 40))])
    o = vo.ttm_squeeze(c + 0.1, c - 0.1, c, 20)
    assert o["squeeze_on"][:80].any()
    assert o["fire"].any()
    assert not (o["fire"] & o["squeeze_on"]).any()


def test_chandelier_flips():
    c = np.concatenate([np.linspace(100, 130, 40), np.linspace(130, 90, 40)])
    o = vo.chandelier(c + 1, c - 1, c, 10, 2.0)
    assert o["long_trend"][35] and o["short_trend"][-1]
    assert o["sell"].sum() == 1


def test_ichimoku_cloud_uses_past_only():
    h = np.arange(100, dtype=float) + 1
    l = np.arange(100, dtype=float)
    o = t2.ichimoku(h, l, (h + l) / 2, 9, 26, 52, 26)
    assert o["tenkan"][8] == pytest.approx((9 + 0) / 2)
    assert np.isnan(o["cloud_b"][:51 + 25]).all()
    assert o["cloud_b"][76] == pytest.approx((52 + 0) / 2)  # spanB computed at bar 51


def test_hull_rising_on_ramp():
    o = t2.hull(np.arange(1, 200, dtype=float), 21)
    assert o["rising"][60:].all() and not o["falling"].any()


def test_ssl_flips():
    c = np.concatenate([np.linspace(10, 20, 30), np.linspace(20, 5, 30)])
    o = t2.ssl_channel(c + 0.5, c - 0.5, c, 5)
    assert o["bull"][25] and o["bear"][-1]
    assert o["flip_down"].sum() == 1


def test_utbot_buy_sell():
    c = np.concatenate([np.linspace(120, 100, 20), np.linspace(100, 140, 30), np.linspace(140, 90, 40)])
    o = t2.ut_bot(c + 1, c - 1, c, 2.0, 5)
    assert o["buy"].any() and o["sell"].any()
    assert o["sell"][np.flatnonzero(o["buy"])[0]:].any()  # a sell follows the first buy


def test_range_filter_constant_then_trend():
    c = np.concatenate([np.full(50, 100.0), np.linspace(100, 160, 80)])
    o = t2.range_filter(c, 10, 1.0)
    assert o["bull"][-1] and not o["bear"][-1]
    assert o["buy"].sum() >= 1


def test_qqe_cycles():
    c = 100 + 20 * np.sin(np.arange(400) / 12.0)
    o = t2.qqe(c)
    assert o["cross_up"].any() and o["cross_down"].any()
    ok = ~np.isnan(o["trend"])
    assert set(np.unique(o["trend"][ok])) <= {-1.0, 1.0}
    assert not (o["bull"] & o["bear"]).any()


def test_obv_hand():
    c = np.array([1.0, 2, 2, 1])
    v = np.array([10.0, 5, 7, 3])
    assert vl.obv(c, v, 2)["obv"].tolist() == [0, 5, 5, 2]


def test_cmf_hand():
    h = np.array([2.0, 2, 2]); l = np.array([1.0, 1, 1]); c = np.array([2.0, 1, 2]); v = np.array([1.0, 1, 2])
    assert vl.cmf(h, l, c, v, 3)["cmf"][2] == pytest.approx((1 - 1 + 2) / 4)


def test_vwap_daily_reset():
    t = np.array([0, 3_600_000, 86_400_000, 86_400_000 + 3_600_000], dtype=np.int64)
    h = np.array([2.0, 4, 10, 20]); v = np.array([1.0, 1, 1, 3])
    o = vl.vwap(h, h, h, v, t)
    assert o["vwap"].tolist() == pytest.approx([2, 3, 10, 17.5])


def test_anchored_vwap():
    t = np.arange(5, dtype=np.int64) * 1000
    c = np.array([1.0, 2, 3, 4, 5])
    o = vl.anchored_vwap(c, c, c, np.ones(5), t, anchor_time=2000)
    assert np.isnan(o["vwap"][:2]).all()
    assert o["vwap"][2:].tolist() == pytest.approx([3, 3.5, 4])


def test_cvd_proxy_hand():
    o = np.array([1.0, 1]); h = np.array([2.0, 2]); l = np.array([1.0, 1]); c = np.array([2.0, 1])
    r = vl.cvd(o, h, l, c, np.array([4.0, 4]), 2)
    assert r["cvd"].tolist() == [4.0, 0.0]


def _zigzag():
    pts = [10, 20, 12, 25, 15, 30, 14, 8]
    seg = [np.linspace(a, b, 6, endpoint=False) for a, b in zip(pts[:-1], pts[1:])]
    return np.concatenate(seg + [np.array([float(pts[-1])])])


def test_swing_confirmed_only_after_right_bars():
    x = _zigzag()
    o = st.swings(x + 0.5, x - 0.5, 2, 2)
    ph = np.flatnonzero(o["pivot_high"])
    assert len(ph) >= 2
    for t in ph:
        p = t - 2
        assert x[p] > x[p - 2:p + 3].max() - 1e-9 or x[p] == x[p - 2:p + 3].max()
    assert o["pivot_high_price"][ph[0]] == pytest.approx(x[ph[0] - 2] + 0.5)
    assert not o["pivot_high"][:ph[0]].any()


def test_bos_choch_sequence():
    x = _zigzag()
    o = st.bos_choch(x + 0.5, x - 0.5, x, 2, 2)
    assert o["bos_up"].any()
    assert o["choch_down"].any()
    assert o["bos_up"][o["break_up"].argmax()]  # first break is BOS, not CHoCH


def test_fvg_hand():
    h = np.array([10.0, 11, 12, 13, 12.5]); l = np.array([9.0, 10, 11.5, 12.2, 11.0])
    c = np.array([9.5, 10.5, 12, 13, 11.2])
    o = st.fvg(h, l, c)
    assert o["bull_fvg"].tolist() == [False, False, True, True, False]
    assert o["retest_bull"][4]


def test_order_block_bull_on_break():
    # swing high at 12 (bar 3, confirmed bar 4 with left=right=1), then close 14 breaks it at bar 5;
    # last bearish candle before the break is bar 4 (close<open)
    o_ = np.array([9.0, 10, 11, 12, 12, 12.5])
    c_ = np.array([10.0, 11, 12, 11.5, 11.8, 14.0])
    h_ = np.array([10.2, 11.2, 12.2, 12.4, 12.1, 14.1])
    l_ = np.array([8.9, 9.9, 10.9, 11.4, 11.7, 12.4])
    r = st.order_blocks(o_, h_, l_, c_, 1, 1, 10)
    assert r["bull_ob"].tolist() == [False] * 5 + [True]


def test_mayer_and_pi():
    c = np.full(400, 100.0)
    c[-1] = 250.0
    m = cy.mayer(c, 200)
    assert m["overheated"][-1] and not m["overheated"][-2]
    c2 = np.concatenate([np.full(350, 100.0), np.linspace(100, 6000, 60)])
    assert cy.pi_cycle(c2, 111, 350)["top"].any()
