"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Reference: Bailey, Borwein, López de Prado & Zhu (2015), "The Probability of
Backtest Overfitting", Journal of Computational Finance 20(4).

Algorithm
---------
1. Split the T x N matrix of trial returns into S contiguous row blocks.
2. For every combination c of S/2 blocks (C(S, S/2) of them) build the
   in-sample set J (those blocks) and the out-of-sample set J̄ (the rest).
3. Compute the performance metric of every trial on J and on J̄.
4. n* = argmax of the IS metric. ω̄_c = relative OOS rank of n* in (0, 1);
   λ_c = logit(ω̄_c).
5. PBO = fraction of combinations with λ_c <= 0 (the IS-best trial is in the
   worse half OOS).

Fast path: for ``metric="sharpe"`` and ``metric="mean"`` the IS/OOS statistics
are obtained from per-block sums via one matrix product per chunk of
combinations, so C(16, 8) = 12,870 splits on N ~ 1000 trials take seconds.
Any other metric is a callable ``f(sub_matrix) -> (N,)`` and is looped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

__all__ = ["cscv_pbo", "CSCVResult", "sharpe_metric", "mean_metric"]

MetricType = Union[str, Callable[[np.ndarray], np.ndarray]]


def sharpe_metric(sub: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Per-column per-period Sharpe of a (t x N) returns sub-matrix."""
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mu / sd, np.nan)


def mean_metric(sub: np.ndarray) -> np.ndarray:
    return sub.mean(axis=0)


@dataclass
class CSCVResult:
    pbo: float                      # Prob[logit <= 0]
    n_splits: int
    n_trials: int
    n_blocks: int
    logits: np.ndarray              # (n_splits,) λ_c
    oos_rel_rank: np.ndarray        # (n_splits,) ω̄_c in (0, 1)
    is_best_idx: np.ndarray         # (n_splits,) n* per split
    is_metric_best: np.ndarray      # IS metric of n*
    oos_metric_best: np.ndarray     # OOS metric of n*
    degradation_slope: float        # OLS slope of OOS_best on IS_best
    degradation_intercept: float
    degradation_r2: float
    prob_loss: float                # Prob[OOS metric of n* < 0]
    mean_is_best: float
    mean_oos_best: float
    threshold: float
    passed: bool

    def to_dict(self, arrays: bool = False) -> dict:
        d = asdict(self)
        if not arrays:
            for k in ("logits", "oos_rel_rank", "is_best_idx", "is_metric_best", "oos_metric_best"):
                d.pop(k)
        return d


def _combination_matrix(n_blocks: int) -> np.ndarray:
    """Boolean (C(S, S/2) x S) matrix: True where the block is in-sample."""
    half = n_blocks // 2
    combos = np.fromiter(
        (b for c in combinations(range(n_blocks), half) for b in c), dtype=np.int64
    ).reshape(-1, half)
    mat = np.zeros((combos.shape[0], n_blocks), dtype=bool)
    rows = np.repeat(np.arange(combos.shape[0]), half)
    mat[rows, combos.ravel()] = True
    return mat


def _rank_and_logit(oos: np.ndarray, oos_best: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative rank ω̄ = rank / (N + 1) of the IS-best trial's OOS metric, and its logit."""
    n_trials = oos.shape[1]
    # rank in {1..N}: 1 + number of trials strictly worse OOS (ties count as half)
    worse = (oos < oos_best[:, None]).sum(axis=1)
    ties = (oos == oos_best[:, None]).sum(axis=1) - 1  # exclude n* itself
    rank = 1.0 + worse + 0.5 * ties
    omega = rank / (n_trials + 1.0)
    logit = np.log(omega / (1.0 - omega))
    return omega, logit


def cscv_pbo(
    trial_returns,
    n_blocks: int = 16,
    metric: MetricType = "sharpe",
    chunk_size: int = 1024,
    threshold: float = 0.2,
    drop_nan_trials: bool = True,
) -> CSCVResult:
    """Probability of Backtest Overfitting of a family of N trials.

    Parameters
    ----------
    trial_returns : (T x N) array or DataFrame
        Per-period returns of each trial (column), aligned in time (row).
    n_blocks : int
        S, must be even. Default 16 -> C(16, 8) = 12,870 splits.
    metric : "sharpe" | "mean" | callable
        Performance statistic per trial. Callables receive a (t x N) float
        sub-matrix and must return an (N,) vector (higher = better).
    chunk_size : int
        Combinations processed per matrix product (memory control).
    threshold : float
        Pass level: PBO < threshold (plan.md: 0.2).
    """
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer >= 2")
    m = trial_returns.to_numpy(dtype=float) if isinstance(trial_returns, pd.DataFrame) else np.asarray(trial_returns, dtype=float)
    if m.ndim != 2:
        raise ValueError("trial_returns must be a T x N matrix")
    if drop_nan_trials:
        good = ~np.isnan(m).any(axis=0)
        m = m[:, good]
    elif np.isnan(m).any():
        raise ValueError("trial_returns contains NaN")
    t_len, n_trials = m.shape
    if n_trials < 2:
        raise ValueError("need at least 2 trials")
    if t_len < n_blocks:
        raise ValueError("fewer rows than blocks")

    block_bounds = np.linspace(0, t_len, n_blocks + 1).astype(int)
    combo_mat = _combination_matrix(n_blocks)
    n_splits = combo_mat.shape[0]

    is_best_idx = np.empty(n_splits, dtype=np.int64)
    is_best = np.empty(n_splits)
    oos_best = np.empty(n_splits)
    omega = np.empty(n_splits)
    logits = np.empty(n_splits)

    fast = isinstance(metric, str)
    if fast:
        metric_name = metric.lower()
        if metric_name not in ("sharpe", "mean"):
            raise ValueError("string metric must be 'sharpe' or 'mean'")
        # per-block sufficient statistics
        block_n = np.diff(block_bounds).astype(float)                       # (S,)
        block_sum = np.empty((n_blocks, n_trials))
        block_sq = np.empty((n_blocks, n_trials))
        for s in range(n_blocks):
            seg = m[block_bounds[s]:block_bounds[s + 1]]
            block_sum[s] = seg.sum(axis=0)
            block_sq[s] = np.einsum("ij,ij->j", seg, seg)
        tot_n = block_n.sum()
        tot_sum = block_sum.sum(axis=0)
        tot_sq = block_sq.sum(axis=0)

        def stats_from(cm: np.ndarray):
            cnt = cm @ block_n                                               # (c,)
            s1 = cm @ block_sum                                              # (c, N)
            is_mean = s1 / cnt[:, None]
            oos_cnt = tot_n - cnt
            oos_mean = (tot_sum[None, :] - s1) / oos_cnt[:, None]
            if metric_name == "mean":
                return is_mean, oos_mean
            s2 = cm @ block_sq
            with np.errstate(divide="ignore", invalid="ignore"):
                is_var = (s2 - cnt[:, None] * is_mean**2) / (cnt[:, None] - 1.0)
                oos_var = ((tot_sq[None, :] - s2) - oos_cnt[:, None] * oos_mean**2) / (oos_cnt[:, None] - 1.0)
                is_sr = np.where(is_var > 0, is_mean / np.sqrt(np.maximum(is_var, 0)), np.nan)
                oos_sr = np.where(oos_var > 0, oos_mean / np.sqrt(np.maximum(oos_var, 0)), np.nan)
            return is_sr, oos_sr

        for start in range(0, n_splits, chunk_size):
            cm = combo_mat[start:start + chunk_size].astype(float)
            is_m, oos_m = stats_from(cm)
            _fill(start, is_m, oos_m, is_best_idx, is_best, oos_best, omega, logits)
    else:
        row_in_block = np.repeat(np.arange(n_blocks), np.diff(block_bounds))
        for c in range(n_splits):
            is_rows = combo_mat[c][row_in_block]
            is_m = np.asarray(metric(m[is_rows]), dtype=float)[None, :]
            oos_m = np.asarray(metric(m[~is_rows]), dtype=float)[None, :]
            _fill(c, is_m, oos_m, is_best_idx, is_best, oos_best, omega, logits)

    pbo = float(np.mean(logits <= 0.0))
    finite = np.isfinite(is_best) & np.isfinite(oos_best)
    if finite.sum() >= 2 and np.ptp(is_best[finite]) > 0:
        slope, intercept = np.polyfit(is_best[finite], oos_best[finite], 1)
        pred = slope * is_best[finite] + intercept
        ss_res = float(np.sum((oos_best[finite] - pred) ** 2))
        ss_tot = float(np.sum((oos_best[finite] - oos_best[finite].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        slope = intercept = r2 = float("nan")
    prob_loss = float(np.mean(oos_best[finite] < 0.0)) if finite.any() else float("nan")

    return CSCVResult(
        pbo=pbo,
        n_splits=int(n_splits),
        n_trials=int(n_trials),
        n_blocks=int(n_blocks),
        logits=logits,
        oos_rel_rank=omega,
        is_best_idx=is_best_idx,
        is_metric_best=is_best,
        oos_metric_best=oos_best,
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        degradation_r2=float(r2),
        prob_loss=prob_loss,
        mean_is_best=float(np.nanmean(is_best)),
        mean_oos_best=float(np.nanmean(oos_best)),
        threshold=float(threshold),
        passed=bool(pbo < threshold),
    )


def _fill(start, is_m, oos_m, is_best_idx, is_best, oos_best, omega, logits):
    c = is_m.shape[0]
    is_filled = np.where(np.isfinite(is_m), is_m, -np.inf)
    best = is_filled.argmax(axis=1)
    rows = np.arange(c)
    ob = oos_m[rows, best]
    oos_filled = np.where(np.isfinite(oos_m), oos_m, -np.inf)
    om, lg = _rank_and_logit(oos_filled, np.where(np.isfinite(ob), ob, -np.inf))
    sl = slice(start, start + c)
    is_best_idx[sl] = best
    is_best[sl] = is_m[rows, best]
    oos_best[sl] = ob
    omega[sl] = om
    logits[sl] = lg
