"""Milestone M4 acceptance: noise strategies -> PBO ~ 0.5 & DSR fail; planted edge passes."""
import numpy as np
import pandas as pd
import pytest

from btclab.research.stats import cscv_pbo, deflated_sharpe_ratio, run_gauntlet, spa_test, trial_sharpes

PPY = 365  # must match conftest.make_family


def _best(m):
    ts = trial_sharpes(m)
    return int(np.nanargmax(ts)), ts


def test_null_family_dsr_fails(noise_family):
    best, ts = _best(noise_family)
    res = deflated_sharpe_ratio(noise_family[:, best], ts, periods_per_year=PPY)
    assert res.n_trials == 500
    assert res.psr > 0.99          # the best of 500 looks great before deflation...
    assert res.dsr < 0.95          # ...and fails after
    assert not res.passed


def test_null_family_pbo_about_half(noise_family):
    res = cscv_pbo(noise_family, n_blocks=16)
    assert res.n_splits == 12870
    assert 0.3 < res.pbo < 0.7
    assert not res.passed


def test_null_family_spa_not_significant(noise_family):
    res = spa_test(noise_family, 0.0, n_boot=2000, seed=0)
    assert res.spa_pvalue > 0.1
    assert res.rc_pvalue > 0.1
    assert not res.passed


def test_planted_edge_dsr_passes(planted_family):
    best, ts = _best(planted_family)
    assert best == 0
    res = deflated_sharpe_ratio(planted_family[:, 0], ts, periods_per_year=PPY)
    assert 1.5 < res.sr_annualised < 2.5
    assert res.dsr > 0.95 and res.passed


def test_planted_edge_pbo_low(planted_family):
    res = cscv_pbo(planted_family, n_blocks=16)
    assert res.pbo < 0.2 and res.passed
    assert (res.is_best_idx == 0).mean() > 0.8


def test_planted_edge_spa_significant(planted_family):
    res = spa_test(planted_family, 0.0, n_boot=2000, seed=0)
    assert res.best_index == 0
    assert res.spa_pvalue < 0.05 and res.rc_pvalue < 0.05
    assert res.passed


def test_gauntlet_end_to_end(planted_family, noise_family):
    t = planted_family.shape[0]
    idx = pd.date_range("2018-01-01", periods=t, freq="D")
    rng = np.random.default_rng(11)
    bh = pd.Series(rng.standard_normal(t) * 0.01, index=idx)       # zero-drift, low-vol benchmark
    close = 20000 * (1 + bh).cumprod()
    strat = pd.Series(planted_family[:, 0], index=idx)
    folds = [strat.iloc[i:i + 180].to_numpy() for i in range(0, 1080, 180)]
    grid = pd.DataFrame({"a": [10, 12, 14, 16], "b": [1.0, 1.1, 1.2, 1.3], "score": [1.0, 1.2, 1.1, 0.9]})

    rep = run_gauntlet(
        strat,
        trial_returns=planted_family,
        benchmark_returns=bh,
        oos_fold_returns=folds,
        param_grid=grid, param_cols=["a", "b"], score_col="score",
        close=close,
        holdout_returns=strat.iloc[-240:],
        n_is_trades=150, oos_fold_trade_counts=[40, 35, 50, 33, 45, 38],
        max_drawdown=0.25, profit_factor=1.5,
        periods_per_year=PPY,
        criteria={"bootstrap_paths": 2000, "spa_n_boot": 1000},
    )
    names = set(rep.results)
    assert {"wf_oos_sharpe", "dsr", "pbo", "spa_vs_zero", "spa_vs_bh", "bootstrap", "stability", "regimes",
            "min_is_trades", "min_oos_trades", "max_drawdown", "profit_factor"} <= names
    assert rep.n_skipped == 0
    for key in ("dsr", "pbo", "spa_vs_zero", "spa_vs_bh", "bootstrap", "stability", "min_is_trades",
                "max_drawdown", "profit_factor"):
        assert rep.results[key].passed, rep.summary()
    # regimes vs a zero-drift benchmark: executed, integer count of regimes where the strategy lags
    assert rep.results["regimes"].passed is not None
    assert isinstance(rep.results["regimes"].value, int)
    assert rep.holdout is not None and rep.holdout["n_obs"] == 240
    d = rep.to_dict()
    assert d["results"]["pbo"]["status"] == "pass"
    assert "GAUNTLET" in rep.summary()

    # a noise strategy fails the family-level tests and the overall verdict
    nb, _ = _best(noise_family)
    rep_noise = run_gauntlet(pd.Series(noise_family[:, nb], index=idx), trial_returns=noise_family,
                             periods_per_year=PPY, criteria={"bootstrap_paths": 1000, "spa_n_boot": 500})
    assert not rep_noise.results["dsr"].passed
    assert not rep_noise.results["pbo"].passed
    assert not rep_noise.passed
    assert rep_noise.results["stability"].passed is None and rep_noise.n_skipped >= 3
