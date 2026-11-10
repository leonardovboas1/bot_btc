"""Purged + embargoed walk-forward folds over a timestamp index, and holdout cutoff.

Layout (plan.md section C): the last ``holdout_months`` (8) are an untouched
holdout; the remaining history is covered by ``n_folds`` (6) rolling folds of
``train_months`` (18) train / ``test_months`` (6) test.

For each fold (bars are index positions, times are index values):

    [ train ............ ][purge][ embargo ][ test .......... ]
    train_start      train_end_nominal   test_start        test_end

* purge   : the last ``purge_bars`` bars of the nominal train window are dropped
            (set it to the strategy's max lookback so no train bar's indicator
            window overlaps a test bar).
* embargo : a time gap between the nominal train end and the test start.

Folds are anchored at the end: the last fold's test ends at ``end`` (default
= holdout cutoff) and earlier folds step back by ``step_months``. With
``step_months=None`` the step is ``test_months`` when the history is long
enough, otherwise it is shrunk (overlapping test windows) so that all folds fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

__all__ = ["Fold", "walk_forward_folds", "holdout_cutoff", "validate_folds"]

TimedeltaLike = Union[pd.Timedelta, str, int, float]


@dataclass
class Fold:
    fold: int
    train_idx: np.ndarray       # integer positions into the index
    test_idx: np.ndarray
    train_start: pd.Timestamp   # first train bar
    train_end: pd.Timestamp     # last train bar (after purge)
    test_start: pd.Timestamp    # first test bar
    test_end: pd.Timestamp      # last test bar
    purge_bars: int
    embargo: pd.Timedelta

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)

    def __repr__(self) -> str:
        return (
            f"Fold({self.fold}: train {self.train_start.date()}..{self.train_end.date()} "
            f"[{self.n_train}], test {self.test_start.date()}..{self.test_end.date()} [{self.n_test}])"
        )


def _to_timedelta(x: TimedeltaLike) -> pd.Timedelta:
    if isinstance(x, pd.Timedelta):
        return x
    if isinstance(x, (int, float)):
        return pd.Timedelta(days=float(x))
    return pd.Timedelta(x)


def _as_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    if idx.has_duplicates:
        raise ValueError("index has duplicate timestamps")
    return idx


def holdout_cutoff(index, holdout_months: int = 8) -> pd.Timestamp:
    """Timestamp such that ``index >= cutoff`` is the untouched holdout.

    The holdout is the last ``holdout_months`` calendar months before the last
    bar: cutoff = last_timestamp - DateOffset(months=holdout_months).
    """
    idx = _as_index(index)
    return idx[-1] - pd.DateOffset(months=holdout_months)


def walk_forward_folds(
    index,
    n_folds: int = 6,
    train_months: int = 18,
    test_months: int = 6,
    step_months: Optional[float] = None,
    purge_bars: int = 0,
    embargo: TimedeltaLike = "5D",
    end: Optional[pd.Timestamp] = None,
    holdout_months: Optional[int] = 8,
    anchored: bool = False,
) -> List[Fold]:
    """Generate purged, embargoed, end-anchored walk-forward folds.

    Parameters
    ----------
    index : DatetimeIndex-like of bar timestamps (sorted, unique)
    n_folds, train_months, test_months : fold geometry
    step_months : spacing between consecutive test starts; None = auto
    purge_bars : bars dropped from the end of each train window (max lookback)
    embargo : time gap between nominal train end and test start ("5D", Timedelta, days)
    end : exclusive end of the last test window; default = holdout cutoff
    holdout_months : used only to derive ``end`` when ``end`` is None
    anchored : if True, every train window starts at the first bar (expanding)
    """
    idx = _as_index(index)
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    emb = _to_timedelta(embargo)
    if end is None:
        end = holdout_cutoff(idx, holdout_months) if holdout_months else idx[-1] + pd.Timedelta(1, "ns")
    end = pd.Timestamp(end)
    first = idx[0]
    train_off = pd.DateOffset(months=train_months)
    test_off = pd.DateOffset(months=test_months)

    def test_bounds(k: int, step: Optional[int], step_days: float) -> tuple[pd.Timestamp, pd.Timestamp]:
        """[test_start, test_end) of fold k. Whole-month steps are laid out with
        month offsets from ``end`` so consecutive folds tile exactly; a fractional
        step (auto-shrunk or user-given) is a day count."""
        back = n_folds - 1 - k
        if step is not None:
            test_end = end - pd.DateOffset(months=step * back)
            test_start = end - pd.DateOffset(months=step * back + test_months)
        else:
            test_end = end - pd.Timedelta(days=step_days * back)
            test_start = test_end - test_off
        return test_start, test_end

    def earliest_train_start(step: Optional[int], step_days: float) -> pd.Timestamp:
        return test_bounds(0, step, step_days)[0] - emb - train_off

    step_int: Optional[int] = None
    step_days = 0.0
    if step_months is None:
        if n_folds == 1 or earliest_train_start(test_months, 0.0) >= first:
            step_int = int(test_months)
        else:
            # shrink the step so the first fold's train window starts at the first bar
            last_train_start = (end - test_off) - emb - train_off
            step_days = (last_train_start - first).days / (n_folds - 1)
            if step_days <= 0:
                raise ValueError(
                    f"history {first.date()}..{end.date()} too short for train {train_months}m + test {test_months}m"
                )
    else:
        if float(step_months).is_integer():
            step_int = int(step_months)
        else:
            step_days = float(step_months) * 30.4375
        est = earliest_train_start(step_int, step_days)
        if est < first:
            raise ValueError(
                f"history starts {first.date()} but fold 0 needs train data from {est.date()}; "
                "reduce n_folds/step or lengths"
            )

    folds: List[Fold] = []
    for k in range(n_folds):
        test_start, test_end = test_bounds(k, step_int, step_days)
        train_end_nominal = test_start - emb
        train_start = first if anchored else train_end_nominal - train_off
        if train_start < first:
            raise ValueError(f"fold {k}: train window starts before the first bar")
        train_pos = np.flatnonzero((idx >= train_start) & (idx < train_end_nominal))
        if purge_bars > 0:
            train_pos = train_pos[:-purge_bars] if purge_bars < train_pos.size else train_pos[:0]
        test_pos = np.flatnonzero((idx >= test_start) & (idx < test_end))
        if train_pos.size == 0 or test_pos.size == 0:
            raise ValueError(f"fold {k}: empty train or test window")
        folds.append(
            Fold(
                fold=k,
                train_idx=train_pos,
                test_idx=test_pos,
                train_start=idx[train_pos[0]],
                train_end=idx[train_pos[-1]],
                test_start=idx[test_pos[0]],
                test_end=idx[test_pos[-1]],
                purge_bars=int(purge_bars),
                embargo=emb,
            )
        )
    return folds


def validate_folds(folds: List[Fold], index, holdout_start: Optional[pd.Timestamp] = None) -> None:
    """Raise AssertionError if any fold violates purge/embargo/holdout separation."""
    idx = _as_index(index)
    for f in folds:
        assert f.train_idx.max() < f.test_idx.min(), f"fold {f.fold}: train after test"
        gap_bars = int(f.test_idx.min() - f.train_idx.max() - 1)
        assert gap_bars >= f.purge_bars, f"fold {f.fold}: purge gap {gap_bars} < {f.purge_bars}"
        assert f.test_start - f.train_end >= f.embargo, f"fold {f.fold}: embargo violated"
        assert not np.intersect1d(f.train_idx, f.test_idx).size, f"fold {f.fold}: overlap"
        if holdout_start is not None:
            assert idx[f.test_idx].max() < holdout_start, f"fold {f.fold}: test touches holdout"
            assert idx[f.train_idx].max() < holdout_start, f"fold {f.fold}: train touches holdout"
