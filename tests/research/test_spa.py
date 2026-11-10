import numpy as np
import pandas as pd
import pytest
from scipy import stats

from btclab.research.stats.bootstrap import stationary_bootstrap_indices
from btclab.research.stats.spa import long_run_variance, spa_test


def test_stationary_bootstrap_indices_structure():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 200, mean_block=10, rng=rng)
    assert idx.shape == (200, 500)
    assert idx.min() >= 0 and idx.max() < 500
    # consecutive indices continue the block unless a new block starts
    step = np.diff(idx, axis=1) % 500
    frac_continue = np.mean(step == 1)
    assert 0.85 < frac_continue < 0.95   # ~ 1 - 1/mean_block (+ chance hits)


def test_long_run_variance_iid_equals_variance():
    rng = np.random.default_rng(1)
    d = rng.standard_normal((5000, 3)) * 0.01
    om2 = long_run_variance(d, mean_block=1.0)   # q = 1 -> kernel weights vanish
    np.testing.assert_allclose(om2, d.var(axis=0), rtol=1e-9)
    om2b = long_run_variance(d, mean_block=10.0)
    np.testing.assert_allclose(om2b, d.var(axis=0), rtol=0.15)


def test_long_run_variance_ar1_is_larger():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(20000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = 0.5 * x[t - 1] + e[t]
    om2 = long_run_variance(x[:, None], mean_block=50.0)[0]
    # true long-run variance of AR(1): var * (1+phi)/(1-phi) = (1/(1-phi^2)) * 3 = 4
    assert 3.0 < om2 < 5.0


def test_single_strategy_rc_matches_t_test():
    rng = np.random.default_rng(7)
    r = rng.standard_normal(2000) * 0.01 + 0.0005
    res = spa_test(r, 0.0, mean_block=1.0, n_boot=4000, seed=1)
    p_t = stats.ttest_1samp(r, 0, alternative="greater").pvalue
    assert res.rc_pvalue == pytest.approx(p_t, abs=0.03)
    assert res.spa_pvalue == pytest.approx(p_t, abs=0.03)
    assert res.n_strategies == 1 and res.best_index == 0


def test_strong_strategy_rejects_and_series_benchmark():
    rng = np.random.default_rng(3)
    t = 1500
    bh = rng.standard_normal(t) * 0.03 + 0.0005
    fam = rng.standard_normal((t, 50)) * 0.02
    fam[:, 7] = bh + 0.004   # beats B&H by 40 bp/day
    res = spa_test(pd.DataFrame(fam), pd.Series(bh), n_boot=1000, seed=0)
    assert res.best_index == 7
    assert res.spa_pvalue < 0.01 and res.rc_pvalue < 0.01
    assert res.passed
    assert res.spa_pvalue_lower <= res.spa_pvalue <= res.spa_pvalue_upper


def test_poor_strategies_do_not_inflate_consistent_pvalue():
    rng = np.random.default_rng(4)
    t = 1500
    fam = rng.standard_normal((t, 40)) * 0.02
    fam[:, :30] -= 0.01   # 30 very poor strategies
    res = spa_test(fam, 0.0, n_boot=1000, seed=0)
    assert res.n_poor == 30
    # upper p-value treats them as zero-mean and is >= the consistent one
    assert res.spa_pvalue_upper >= res.spa_pvalue
