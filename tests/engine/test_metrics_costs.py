import numpy as np
import pandas as pd
import pytest

from btclab.engine.costs import BINANCE_PERP_TAKER, HYPERLIQUID_TAKER, PRESETS, SPOT, USER_TV_DEFAULT
from btclab.engine.metrics import compute_metrics, daily_returns, max_drawdown, yearly_returns
from btclab.engine.sim import simulate

H = 3_600_000  # one hour in ms


def hourly_ts(n, start="2021-12-30"):
    t0 = pd.Timestamp(start).value // 10**6
    return t0 + np.arange(n, dtype=np.int64) * H


def test_daily_returns_resample():
    n = 72  # 3 full UTC days
    eq = np.ones(n)
    eq[24:48] = 1.1   # day 2 ends at 1.1
    eq[48:] = 1.21    # day 3 ends at 1.21
    eq[30] = 0.5      # intraday dip must not affect daily closes
    dr = daily_returns(eq, hourly_ts(n))
    assert len(dr) == 3
    np.testing.assert_allclose(dr.to_numpy(), [0.0, 0.1, 0.1])
    assert dr.index[0] == pd.Timestamp("2021-12-30")


def test_daily_returns_accept_datetime_index():
    ts = pd.to_datetime(hourly_ts(48), unit="ms")
    dr = daily_returns(np.linspace(1, 2, 48), ts)
    assert len(dr) == 2


def test_max_drawdown_hand():
    eq = np.array([1.0, 1.2, 0.9, 1.0, 1.3, 1.1, 1.3])
    mdd, peak_i, trough_i, dur = max_drawdown(eq)
    assert mdd == pytest.approx(0.9 / 1.2 - 1)
    assert peak_i == 1 and trough_i == 2
    assert dur == 2  # bars 2,3 under water after the 1.2 peak


def test_yearly_returns():
    n = 72  # 2021-12-30 .. 2022-01-01
    eq = np.ones(n)
    eq[48:] = 1.5  # 2021 closes at 1.0, 2022 bars at 1.5
    yr = yearly_returns(eq, hourly_ts(n))
    assert yr == {2021: pytest.approx(0.0), 2022: pytest.approx(0.5)}


def test_compute_metrics_on_sim_result():
    n = 24 * 400
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    o = np.roll(c, 1)
    o[0] = c[0]
    el = rng.random(n) < 0.01
    xl = rng.random(n) < 0.05
    es = rng.random(n) < 0.01
    xs = rng.random(n) < 0.05
    r = simulate(o, np.maximum(o, c), np.minimum(o, c), c, el, es, xl, xs)
    m = compute_metrics(r.equity, r.trades, hourly_ts(n, "2022-01-01"))
    assert m["n_trades"] == r.n_trades == m["n_long"] + m["n_short"]
    assert m["total_return"] == pytest.approx(r.equity[-1] - 1.0)
    assert m["years"] == pytest.approx((n - 1) / 24 / 365)
    assert m["cagr"] == pytest.approx(r.equity[-1] ** (1 / m["years"]) - 1)
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert 0.0 <= m["win_rate"] <= 1.0
    assert 0.0 < m["exposure"] < 1.0
    assert m["n_days"] == 400
    assert m["max_drawdown_days"] * 24 >= m["max_drawdown_bars"] - 1
    assert set(m["yearly_returns"]) == {2022, 2023}
    assert np.isfinite(m["sharpe"]) and np.isfinite(m["sortino"])
    # sharpe recomputed by hand from daily returns
    dr = daily_returns(r.equity, hourly_ts(n, "2022-01-01"))
    assert m["sharpe"] == pytest.approx(dr.mean() / dr.std(ddof=1) * np.sqrt(365))
    pnl = r.trades["pnl"]
    assert m["profit_factor"] == pytest.approx(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum())
    assert m["expectancy_bp"] == pytest.approx(r.trades["pnl_pct"].mean() * 1e4)


def test_metrics_no_trades():
    n = 48
    m = compute_metrics(np.ones(n), np.empty(0, dtype=simulate(np.ones(2), np.ones(2), np.ones(2), np.ones(2)).trades.dtype), hourly_ts(n))
    assert m["n_trades"] == 0 and m["exposure"] == 0.0 and m["total_return"] == 0.0
    assert np.isnan(m["profit_factor"]) and np.isnan(m["sharpe"])


def test_cost_presets():
    assert BINANCE_PERP_TAKER.fee_rate == 0.0005
    assert USER_TV_DEFAULT.fee_rate == 0.001
    assert SPOT.fee_rate == 0.001 and SPOT.allow_short is False
    assert HYPERLIQUID_TAKER.fee_rate == 0.00045
    assert set(PRESETS) == {"binance_perp_taker", "user_tv_default", "spot", "hyperliquid_taker"}
    kw = SPOT.sim_kwargs()
    assert kw == {"fee_rate": 0.001, "slippage_bp": 2.0, "allow_short": False}
    o = np.full(3, 100.0)
    r = simulate(o, o, o, o, entry_short=[1, 0, 0], **kw)
    assert r.n_trades == 0
