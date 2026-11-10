import numpy as np
import pytest

from btclab.research.stats.bootstrap import block_bootstrap, circular_block_indices, resample_counts


def test_circular_indices_and_counts():
    rng = np.random.default_rng(0)
    idx = circular_block_indices(100, 50, 7, rng)
    assert idx.shape == (50, 100)
    assert ((np.diff(idx, axis=1) % 100 == 1).mean()) > 0.8
    counts = resample_counts(idx, 100)
    assert counts.shape == (50, 100)
    assert np.all(counts.sum(axis=1) == 100)


def test_block_bootstrap_distribution_and_percentiles():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(1000) * 0.02 + 0.001
    res = block_bootstrap(r, n_paths=3000, periods_per_year=365, seed=0)
    assert res.final_equity.shape == (3000,)
    assert np.all(res.max_drawdown >= 0) and np.all(res.max_drawdown <= 1)
    assert res.percentiles.loc[5, "final_equity"] < res.percentiles.loc[50, "final_equity"] < res.percentiles.loc[95, "final_equity"]
    assert res.pct("total_return", 50) == pytest.approx(np.median(res.total_return))
    # bootstrapped median final equity is near the observed one
    assert abs(np.log(res.pct("final_equity", 50)) - np.log(res.observed_final_equity)) < 1.0
    assert 0 <= res.prob_loss <= 1
    assert res.observed_sharpe == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(365))


def test_methods_and_trade_sequences():
    rng = np.random.default_rng(2)
    trades = rng.standard_normal(150) * 0.05 + 0.01
    a = block_bootstrap(trades, n_paths=2000, method="stationary", block_length=5, seed=3)
    b = block_bootstrap(trades, n_paths=2000, method="circular", block_length=5, seed=3)
    assert a.method == "stationary" and b.method == "circular"
    assert abs(a.pct("total_return", 50) - b.pct("total_return", 50)) < 0.5
    with pytest.raises(ValueError):
        block_bootstrap(trades, n_paths=10, method="nope")


def test_losing_strategy_has_negative_fifth_percentile():
    rng = np.random.default_rng(5)
    r = rng.standard_normal(800) * 0.02 - 0.001
    res = block_bootstrap(r, n_paths=2000, seed=0)
    assert res.pct("total_return", 5) < 0
    assert res.prob_loss > 0.5
