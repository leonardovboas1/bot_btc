import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.a7.portfolio import backtest, metrics  # noqa: E402


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def test_causality_target_decided_at_t_earns_t_plus_1():
    ix = _idx(6)
    ret = pd.DataFrame({"a": [0.0, 0.10, 0.0, 0.0, 0.0, 0.0]}, ix)
    tgt = pd.DataFrame({"a": [1.0] * 6}, ix)
    r = backtest(ret, tgt, 0.0).returns
    assert r.iloc[1] == 0.1 or abs(r.iloc[1] - 0.1) < 1e-12   # weight set at day 0 close earns day 1
    # a target that "sees" the day-1 return (set at close of day 1) must not earn day 1
    tgt2 = pd.DataFrame({"a": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, ix)
    assert abs(backtest(ret, tgt2, 0.0).returns.iloc[1]) < 1e-12


def test_future_returns_do_not_change_past_pnl():
    rng = np.random.default_rng(0)
    ix = _idx(50)
    ret = pd.DataFrame(rng.normal(0, 0.02, (50, 2)), ix, columns=["a", "b"])
    tgt = pd.DataFrame(np.sign(ret.rolling(3).mean()).fillna(0), ix)
    base = backtest(ret, tgt, 0.001).returns
    ret2 = ret.copy()
    ret2.iloc[30:] = rng.normal(0, 0.5, (20, 2))
    tgt2 = pd.DataFrame(np.sign(ret2.rolling(3).mean()).fillna(0), ix)
    alt = backtest(ret2, tgt2, 0.001).returns
    assert np.allclose(base.iloc[:30], alt.iloc[:30])


def test_cost_math_and_drift():
    ix = _idx(4)
    ret = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0]}, ix)
    tgt = pd.DataFrame({"a": [1.0, np.nan, 0.0, 0.0]}, ix)
    res = backtest(ret, tgt, 0.001)
    # day 1: buy 1.0 -> cost 0.001; day 2 hold (NaN), no cost; day 3 sell -> cost 0.001*? (target 0 set day 2)
    assert abs(res.cost.iloc[1] - 0.001) < 1e-12
    assert res.cost.iloc[2] == 0.0 or abs(res.cost.iloc[2]) < 1e-12
    assert abs(res.cost.iloc[3] - 0.001 * (1 - 0.001) / (1 - 0.001)) < 1e-5
    assert abs(res.returns.iloc[1] + 0.001) < 1e-12


def test_funding_sign_and_gross_cap():
    ix = _idx(3)
    ret = pd.DataFrame({"p": [0.0] * 3, "q": [0.0] * 3}, ix)
    fund = pd.DataFrame({"p": [0.001] * 3, "q": [0.001] * 3}, ix)
    tgt = pd.DataFrame({"p": [1.0, 1.0, 1.0], "q": [-3.0, -3.0, -3.0]}, ix)
    res = backtest(ret, tgt, 0.0, funding=fund, gross_cap=2.0)
    w = res.weights.iloc[1]
    assert abs(np.abs(w).sum() - 2.0) < 1e-12
    # scaled to p=0.5, q=-1.5: long pays 0.5*0.001, short receives 1.5*0.001 -> +0.001
    assert abs(res.returns.iloc[1] - 0.001) < 1e-12


def test_toy_two_asset_buy_and_hold():
    ix = _idx(3)
    ret = pd.DataFrame({"a": [0.0, 0.10, 0.0], "b": [0.0, -0.10, 0.0]}, ix)
    tgt = pd.DataFrame({"a": [0.5] * 3, "b": [0.5] * 3}, ix)
    res = backtest(ret, tgt, [0.0, 0.0])
    assert abs(res.returns.iloc[1]) < 1e-12
    # drifted weights after day 1: a=0.55/1, b=0.45/1 -> daily rebalance trades 0.05 each leg on day 2
    assert abs(res.turnover.iloc[2] - 0.10) < 1e-12
    res2 = backtest(ret, tgt, [0.001, 0.001])
    assert abs(res2.cost.iloc[2] - 0.0001) < 1e-6


def test_metrics_quarters():
    ix = pd.date_range("2024-01-01", periods=365, freq="D", tz="UTC")
    r = pd.Series(0.001, ix)
    m = metrics(r)
    assert m["pos_q"] == 1.0 and m["mdd"] == 0.0
