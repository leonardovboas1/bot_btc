"""Hand-computed, deterministic tests for btclab.engine.sim."""

import time

import numpy as np
import pytest

from btclab.engine.sim import (
    EXIT_END,
    EXIT_REVERSE,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TAKE_PROFIT,
    EXIT_TIME,
    EXIT_TRAIL,
    SimParams,
    simulate,
)

NOCOST = dict(fee_rate=0.0, slippage_bp=0.0)


def flat(n, px=100.0):
    a = np.full(n, float(px))
    return a, a.copy(), a.copy(), a.copy()


def sig(n, *idx):
    s = np.zeros(n, dtype=bool)
    for i in idx:
        s[i] = True
    return s


def test_fill_at_next_open():
    o = np.array([100.0, 110.0, 120.0, 130.0])
    r = simulate(o, o, o, o, entry_long=sig(4, 0), exit_long=sig(4, 1), **NOCOST)
    assert r.n_trades == 1
    t = r.trades[0]
    assert t["entry_idx"] == 1 and t["entry_px"] == 110.0   # signal on 0 -> open[1]
    assert t["exit_idx"] == 2 and t["exit_px"] == 120.0     # signal on 1 -> open[2]
    assert t["exit_reason"] == EXIT_SIGNAL and t["side"] == 1 and t["bars"] == 1
    np.testing.assert_allclose(r.equity, [1.0, 1.0, 120 / 110, 120 / 110])
    assert list(r.position) == [0, 1, 0, 0]


def test_signal_on_last_bar_never_fills():
    o = np.array([100.0, 110.0, 120.0])
    r = simulate(o, o, o, o, entry_long=sig(3, 2), **NOCOST)
    assert r.n_trades == 0
    np.testing.assert_array_equal(r.equity, 1.0)


def test_fee_round_trip():
    o, h, l, c = flat(5)
    r = simulate(o, h, l, c, entry_long=sig(5, 0), exit_long=sig(5, 2), fee_rate=0.001, slippage_bp=0.0)
    t = r.trades[0]
    # notional 1.0 -> qty 0.01; entry fee 0.001, exit fee 0.01*100*0.001 = 0.001
    assert t["fees"] == pytest.approx(0.002)
    assert t["pnl"] == pytest.approx(-0.002)
    assert t["pnl_pct"] == pytest.approx(-0.002)
    np.testing.assert_allclose(r.equity, [1.0, 0.999, 0.999, 0.998, 0.998])


def test_slippage_bp_and_extra():
    o, h, l, c = flat(4)
    extra = np.array([0.0, 0.001, 0.0, 0.0])  # 10 bp extra on the fill bar
    r = simulate(o, h, l, c, entry_long=sig(4, 0), slip_extra=extra, fee_rate=0.0, slippage_bp=5.0)
    assert r.trades[0]["entry_px"] == pytest.approx(100.0 * (1 + 0.0005 + 0.001))
    r2 = simulate(o, h, l, c, entry_short=sig(4, 0), fee_rate=0.0, slippage_bp=5.0)
    assert r2.trades[0]["entry_px"] == pytest.approx(100.0 * (1 - 0.0005))


def test_stop_before_take_profit_worst_case():
    o = np.array([100.0, 100.0, 100.0])
    h = np.array([100.0, 110.0, 100.0])  # bar 1 touches both TP (105) and SL (95)
    l = np.array([100.0, 90.0, 100.0])
    r = simulate(o, h, l, o, entry_long=sig(3, 0), sl_pct=0.05, tp_pct=0.05, **NOCOST)
    t = r.trades[0]
    assert t["exit_reason"] == EXIT_STOP
    assert t["exit_idx"] == 1 and t["exit_px"] == pytest.approx(95.0)
    assert t["pnl_pct"] == pytest.approx(-0.05)
    np.testing.assert_allclose(r.equity, [1.0, 0.95, 0.95])
    assert list(r.position) == [0, 0, 0]  # closed intrabar, flat at the close


def test_take_profit_hits_when_stop_does_not():
    o = np.array([100.0, 100.0, 100.0])
    h = np.array([100.0, 110.0, 100.0])
    l = np.array([100.0, 99.0, 100.0])
    r = simulate(o, h, l, o, entry_long=sig(3, 0), sl_pct=0.05, tp_pct=0.05, **NOCOST)
    t = r.trades[0]
    assert t["exit_reason"] == EXIT_TAKE_PROFIT and t["exit_px"] == pytest.approx(105.0)


def test_stop_gap_through_fills_at_open():
    o = np.array([100.0, 100.0, 90.0, 90.0])
    h = np.array([100.0, 101.0, 92.0, 90.0])
    l = np.array([100.0, 99.0, 88.0, 90.0])
    r = simulate(o, h, l, o, entry_long=sig(4, 0), sl_pct=0.05, **NOCOST)
    t = r.trades[0]
    assert t["exit_reason"] == EXIT_STOP and t["exit_idx"] == 2
    assert t["exit_px"] == pytest.approx(90.0)  # worse than the 95 level


def test_atr_stop_uses_signal_bar_atr_and_short_side():
    o = np.array([100.0, 100.0, 100.0])
    h = np.array([100.0, 104.0, 100.0])
    l = np.array([100.0, 96.0, 100.0])
    atr = np.array([2.0, 50.0, 50.0])  # only atr[0] (signal bar) may be used
    r = simulate(o, h, l, o, entry_short=sig(3, 0), atr=atr, sl_atr=1.5, **NOCOST)
    t = r.trades[0]
    assert t["side"] == -1 and t["exit_reason"] == EXIT_STOP
    assert t["exit_px"] == pytest.approx(103.0)  # 100 + 1.5 * 2


def test_funding_sign():
    o, h, l, c = flat(5)
    fund = np.array([0.01, 0.0, 0.0001, 0.0, 0.0])  # bar 0 while flat must not count
    r_long = simulate(o, h, l, c, entry_long=sig(5, 0), funding=fund, **NOCOST)
    r_short = simulate(o, h, l, c, entry_short=sig(5, 0), funding=fund, **NOCOST)
    assert r_long.trades[0]["funding"] == pytest.approx(1e-4)    # long pays positive rate
    assert r_short.trades[0]["funding"] == pytest.approx(-1e-4)  # short receives
    np.testing.assert_allclose(r_long.equity, [1, 1, 1 - 1e-4, 1 - 1e-4, 1 - 1e-4])
    np.testing.assert_allclose(r_short.equity, [1, 1, 1 + 1e-4, 1 + 1e-4, 1 + 1e-4])
    assert r_long.trades[0]["pnl"] == pytest.approx(-1e-4)


def test_reverse_on_opposite_signal():
    o = np.array([100.0, 100.0, 110.0, 120.0, 110.0, 100.0])
    n = len(o)
    r = simulate(o, o, o, o, entry_long=sig(n, 0), entry_short=sig(n, 2), reverse_on_signal=True, **NOCOST)
    assert r.n_trades == 2
    a, b = r.trades
    assert a["side"] == 1 and a["exit_idx"] == 3 and a["exit_px"] == 120.0 and a["exit_reason"] == EXIT_REVERSE
    assert b["side"] == -1 and b["entry_idx"] == 3 and b["entry_px"] == 120.0 and b["exit_reason"] == EXIT_END
    assert list(r.position) == [0, 1, 1, -1, -1, -1]
    # long: 1.0 -> 1.2 ; short notional 1.2 at 120, close 100 -> +0.2
    np.testing.assert_allclose(r.equity, [1.0, 1.0, 1.1, 1.2, 1.3, 1.4])
    # without the flag the opposite entry is ignored (no pyramiding, no reverse)
    r2 = simulate(o, o, o, o, entry_long=sig(n, 0), entry_short=sig(n, 2), reverse_on_signal=False, **NOCOST)
    assert r2.n_trades == 1 and list(r2.position) == [0, 1, 1, 1, 1, 1]
    assert r2.equity[-1] == pytest.approx(1.0)


def test_allow_short_false_ignores_short_entries():
    o, h, l, c = flat(4)
    r = simulate(o, h, l, c, entry_short=sig(4, 0), allow_short=False, **NOCOST)
    assert r.n_trades == 0


def test_no_pyramiding():
    o = np.array([100.0, 100.0, 100.0, 100.0])
    r = simulate(o, o, o, o, entry_long=sig(4, 0, 1, 2), **NOCOST)
    assert r.n_trades == 1 and r.trades[0]["entry_idx"] == 1


def test_zero_signals_flat_equity():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    r = simulate(c, c * 1.01, c * 0.99, c, init_cash=5.0)
    assert r.n_trades == 0
    np.testing.assert_array_equal(r.equity, 5.0)
    np.testing.assert_array_equal(r.position, 0)


def test_buy_and_hold_equality():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 1000)))
    o = np.roll(c, 1)
    o[0] = c[0]
    fee = 0.0005
    r = simulate(o, np.maximum(o, c), np.minimum(o, c), c, entry_long=sig(1000, 0), fee_rate=fee, slippage_bp=0.0)
    assert r.n_trades == 1
    t = r.trades[0]
    assert t["exit_reason"] == EXIT_END and t["exit_idx"] == 999 and t["exit_px"] == c[-1]
    assert t["fees"] == pytest.approx(fee)  # no exit fee on the open position
    assert r.equity[0] == 1.0
    np.testing.assert_allclose(r.equity[1:], c[1:] / o[1] - fee, rtol=1e-12)
    assert r.position[1:].all()


def test_time_stop():
    o, h, l, c = flat(6)
    r = simulate(o, h, l, c, entry_long=sig(6, 0), time_stop=2, **NOCOST)
    t = r.trades[0]
    assert t["entry_idx"] == 1 and t["exit_idx"] == 3 and t["exit_reason"] == EXIT_TIME


def test_trailing_stop_pct_ratchets_from_previous_bars():
    o = np.array([100.0, 100.0, 100.0, 115.0, 115.0])
    h = np.array([100.0, 100.0, 120.0, 116.0, 115.0])
    l = np.array([100.0, 100.0, 100.0, 105.0, 115.0])
    r = simulate(o, h, l, o, entry_long=sig(5, 0), trail_pct=0.10, **NOCOST)
    t = r.trades[0]
    # bar 2: level still 90 during the bar, ratchets to 108 after; bar 3 low 105 hits 108
    assert t["exit_reason"] == EXIT_TRAIL and t["exit_idx"] == 3
    assert t["exit_px"] == pytest.approx(108.0)


def test_leverage_and_size_frac():
    o = np.array([100.0, 100.0, 110.0])
    r = simulate(o, o, o, o, entry_long=sig(3, 0), size_frac=0.5, leverage=4.0, **NOCOST)
    # notional 2.0 -> +10% move = +0.2
    assert r.equity[-1] == pytest.approx(1.2)


def test_bankruptcy_clamps_to_zero():
    o = np.array([100.0, 100.0, 50.0, 60.0, 70.0])
    r = simulate(o, o, o, o, entry_long=sig(5, 0), leverage=3.0, exit_long=sig(5, 2), **NOCOST)
    assert r.equity[2] == 0.0 and r.equity[-1] == 0.0


def test_sim_params_overrides():
    o, h, l, c = flat(3)
    p = SimParams(fee_rate=0.01, slippage_bp=0.0)
    r = simulate(o, h, l, c, entry_long=sig(3, 0), params=p, fee_rate=0.0)
    assert r.trades[0]["fees"] == 0.0 and r.params.fee_rate == 0.0


def test_performance_200k_bars():
    n = 200_000
    rng = np.random.default_rng(42)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.roll(c, 1)
    o[0] = c[0]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    atr = np.full(n, 0.5)
    fund = np.where(np.arange(n) % 8 == 0, 0.0001, 0.0)
    el = rng.random(n) < 0.02
    es = rng.random(n) < 0.02
    xl = rng.random(n) < 0.05
    xs = rng.random(n) < 0.05
    kw = dict(atr=atr, funding=fund, sl_atr=3.0, tp_atr=6.0, trail_atr=4.0, time_stop=48, reverse_on_signal=True)
    simulate(o, h, l, c, el, es, xl, xs, **kw)  # warm-up / cache load
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        r = simulate(o, h, l, c, el, es, xl, xs, **kw)
        best = min(best, time.perf_counter() - t0)
    print(f"\n200k bars: best {best * 1e3:.1f} ms, {r.n_trades} trades")
    assert r.n_trades > 1000
    assert np.isfinite(r.equity).all()
    assert best < 0.2
