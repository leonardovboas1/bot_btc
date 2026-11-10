import numpy as np
import pandas as pd
import pytest

from btclab.research.stats.regimes import REGIME_LABELS, label_regimes, regime_performance


def _close(n, seed=0):
    rng = np.random.default_rng(seed)
    # vol clustering so terciles are meaningful
    vol = 0.01 * np.exp(np.cumsum(rng.standard_normal(n) * 0.05) * 0.5)
    vol = np.clip(vol, 0.003, 0.1)
    r = rng.standard_normal(n) * vol + 0.0003
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.Series(20000 * np.cumprod(1 + r), index=idx)


def test_labels_are_causal_when_future_is_appended():
    full = _close(1500)
    labels_full = label_regimes(full)
    for cut in (400, 700, 1000, 1499):
        labels_part = label_regimes(full.iloc[:cut])
        pd.testing.assert_series_equal(labels_part, labels_full.iloc[:cut])


def test_label_values_and_warmup():
    close = _close(1200)
    lab = label_regimes(close, sma_window=200, vol_window=20)
    assert lab.iloc[:200].isna().all()
    valid = lab.dropna()
    assert len(valid) > 900
    assert set(valid.unique()) <= set(REGIME_LABELS)
    # both trend signs and all three vol buckets appear
    assert {s.split("_")[0] for s in valid.unique()} == {"up", "down"}
    assert {s.split("_")[1] for s in valid.unique()} == {"lowvol", "midvol", "highvol"}


def test_terciles_are_expanding_not_full_sample():
    close = _close(1200, seed=3)
    lab = label_regimes(close)
    # with full-sample terciles each bucket would hold ~1/3 of the bars; with
    # expanding terciles the early buckets are usually unbalanced. We check the
    # causality property directly instead: a late high-vol spike must not
    # relabel early bars.
    spiked = close.copy()
    spiked.iloc[1000:] = spiked.iloc[1000:] * np.cumprod(1 + np.random.default_rng(1).standard_normal(200) * 0.08)
    lab2 = label_regimes(spiked)
    pd.testing.assert_series_equal(lab.iloc[:1000], lab2.iloc[:1000])


def test_regime_performance_breakdown():
    close = _close(1000, seed=5)
    reg = label_regimes(close)
    bh = close.pct_change()
    strat = bh * 0.5 + 0.0005
    perf = regime_performance(strat, reg, bh, periods_per_year=365)
    assert "all" in perf.index
    assert perf.loc["all", "n"] == reg.notna().sum()
    assert {"mean", "sharpe", "total_return", "max_drawdown", "bench_mean", "excess", "worse_than_benchmark"} <= set(perf.columns)
    assert perf.drop(index="all")["n"].sum() == perf.loc["all", "n"]
