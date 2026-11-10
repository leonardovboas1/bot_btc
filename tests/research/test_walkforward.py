import numpy as np
import pandas as pd
import pytest

from btclab.research.stats.walkforward import holdout_cutoff, validate_folds, walk_forward_folds


@pytest.fixture
def daily_index():
    return pd.date_range("2017-08-17", "2026-11-09", freq="D", tz="UTC")


def test_holdout_cutoff_reserves_last_8_months(daily_index):
    cut = holdout_cutoff(daily_index, 8)
    assert cut == daily_index[-1] - pd.DateOffset(months=8)
    holdout = daily_index[daily_index >= cut]
    assert 240 <= len(holdout) <= 246


def test_default_folds_respect_purge_embargo_and_holdout(daily_index):
    purge = 200
    folds = walk_forward_folds(daily_index, purge_bars=purge, embargo="5D")
    cut = holdout_cutoff(daily_index, 8)
    assert len(folds) == 6
    validate_folds(folds, daily_index, holdout_start=cut)
    for f in folds:
        # gap between last train bar and first test bar covers purge + embargo
        gap_bars = f.test_idx.min() - f.train_idx.max() - 1
        assert gap_bars >= purge
        assert f.test_start - f.train_end >= pd.Timedelta("5D")
        assert daily_index[f.test_idx].max() < cut
        assert daily_index[f.train_idx].max() < cut
        # 6-month test, 18-month train (546-550 calendar days) minus purge
        assert 180 <= f.n_test <= 184
        assert 546 - purge <= f.n_train <= 550 - purge
    # consecutive folds roll forward by the test length and tests do not overlap
    for a, b in zip(folds, folds[1:]):
        assert b.test_start > a.test_end
        assert b.test_idx.min() == a.test_idx.max() + 1
    # last fold ends right before the holdout
    assert daily_index[folds[-1].test_idx.max() + 1] >= cut


def test_step_shrinks_when_history_is_short():
    idx = pd.date_range("2022-01-01", "2026-11-09", freq="D")   # ~58 months -> 50 usable
    folds = walk_forward_folds(idx, purge_bars=10)
    cut = holdout_cutoff(idx)
    validate_folds(folds, idx, holdout_start=cut)
    assert len(folds) == 6
    assert folds[0].train_idx.min() == 0
    for a, b in zip(folds, folds[1:]):
        assert b.test_start > a.test_start     # overlapping but strictly rolling


def test_explicit_step_and_errors():
    idx = pd.date_range("2015-01-01", "2026-11-09", freq="D")
    folds = walk_forward_folds(idx, step_months=3, n_folds=4, train_months=12, test_months=3, embargo=2)
    validate_folds(folds, idx, holdout_start=holdout_cutoff(idx))
    assert len(folds) == 4
    with pytest.raises(ValueError):
        walk_forward_folds(pd.date_range("2024-01-01", "2026-11-09", freq="D"), step_months=6)
    with pytest.raises(ValueError):
        walk_forward_folds(idx[::-1])


def test_anchored_and_custom_end():
    idx = pd.date_range("2018-01-01", "2026-11-09", freq="D")
    end = pd.Timestamp("2024-01-01")
    folds = walk_forward_folds(idx, anchored=True, end=end, holdout_months=None, purge_bars=5)
    validate_folds(folds, idx, holdout_start=end)
    assert all(f.train_idx.min() == 0 for f in folds)
    assert folds[-1].test_end < end
