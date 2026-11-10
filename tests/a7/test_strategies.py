import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.a7.portfolio import PERP_COST  # noqa: E402
from research.a7.strategies import Panel, c1_targets, c2_targets, configs, run_c3  # noqa: E402


def _close(n=400, k=4, seed=1):
    rng = np.random.default_rng(seed)
    ix = pd.date_range("2022-01-03", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, (n, k)), 0)), ix,
                        columns=list("ABCD")[:k])


def test_c1_c2_targets_are_causal():
    c = _close()
    c2 = c.copy()
    c2.iloc[300:] *= 1.7                       # change the future only
    for f in (lambda x: c1_targets(x, False, 0.4), lambda x: c2_targets(x, 28, True)):
        pd.testing.assert_frame_equal(f(c).iloc[:300], f(c2).iloc[:300])


def test_c1_bounds_and_c2_weights():
    c = _close()
    t = c1_targets(c, True, 0.4)
    assert (t >= 0).all().all() and (t <= 1).all().all()
    w = c2_targets(c, 14, True).dropna(how="all")
    assert (w.index.dayofweek == 6).all()
    assert abs(w.iloc[-1].clip(lower=0).sum() - 0.5) < 1e-9
    assert abs(w.iloc[-1].clip(upper=0).sum() + 0.5) < 1e-9


def test_c3_carry_toy():
    ix = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    px = pd.DataFrame({"A": 100.0}, ix)
    fund = pd.DataFrame({"A": 0.0005}, ix)     # 18% annualised > 10%
    p = Panel(px, px.pct_change(), fund, PERP_COST, spot_close=px.copy())
    r = run_c3(p, 0.10)
    assert abs(r.returns.iloc[-1] - 0.5 * 0.0005) < 1e-6   # price legs cancel, short perp earns funding
    assert run_c3(p, 0.20).returns.abs().max() == 0.0


def test_twelve_configs():
    assert len(configs()) == 12
