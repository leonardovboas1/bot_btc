"""Block bootstrap of return sequences (daily returns or per-trade returns).

Resamplers
----------
* ``stationary_bootstrap_indices``  Politis & Romano (1994): geometric block
  lengths with mean ``mean_block``, wrapping circularly.
* ``circular_block_indices``        Fixed-length circular moving-block bootstrap.

``block_bootstrap`` draws ``n_paths`` (default 10,000) resampled sequences and
returns the distribution of final equity, max drawdown and Sharpe, with
percentiles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "stationary_bootstrap_indices",
    "circular_block_indices",
    "resample_counts",
    "block_bootstrap",
    "BootstrapResult",
]


def stationary_bootstrap_indices(
    n_obs: int, n_reps: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """(n_reps x n_obs) int matrix of stationary-bootstrap resampling indices.

    Each row: start at a uniform index; at every step, with probability
    p = 1/mean_block jump to a new uniform start, otherwise continue to the next
    index (mod n_obs).
    """
    if mean_block <= 0:
        raise ValueError("mean_block must be > 0")
    p = min(1.0, 1.0 / float(mean_block))
    t = np.arange(n_obs)
    new = rng.random((n_reps, n_obs)) < p
    new[:, 0] = True
    starts = rng.integers(0, n_obs, size=(n_reps, n_obs))
    blk_start_t = np.maximum.accumulate(np.where(new, t[None, :], 0), axis=1)
    start_val = np.take_along_axis(starts, blk_start_t, axis=1)
    return (start_val + (t[None, :] - blk_start_t)) % n_obs


def circular_block_indices(
    n_obs: int, n_reps: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_reps x n_obs) int matrix of circular moving-block bootstrap indices."""
    block_length = int(max(1, block_length))
    n_blocks = int(np.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs, size=(n_reps, n_blocks))
    idx = starts[:, :, None] + np.arange(block_length)[None, None, :]
    return idx.reshape(n_reps, -1)[:, :n_obs] % n_obs


def resample_counts(idx: np.ndarray, n_obs: int) -> np.ndarray:
    """How many times each original index was drawn, per row: (n_reps x n_obs)."""
    n_reps = idx.shape[0]
    flat = idx + (np.arange(n_reps) * n_obs)[:, None]
    return np.bincount(flat.ravel(), minlength=n_reps * n_obs).reshape(n_reps, n_obs).astype(float)


@dataclass
class BootstrapResult:
    n_paths: int
    n_obs: int
    method: str
    block_length: float
    final_equity: np.ndarray       # multiple of starting equity
    total_return: np.ndarray       # final_equity - 1
    max_drawdown: np.ndarray       # positive fraction, e.g. 0.25 = -25 %
    sharpe: np.ndarray             # annualised if periods_per_year given
    percentiles: pd.DataFrame      # rows: percentiles, cols: the three stats
    prob_loss: float               # P[total_return < 0]
    observed_final_equity: float
    observed_max_drawdown: float
    observed_sharpe: float

    def pct(self, stat: str, q: float) -> float:
        return float(self.percentiles.loc[q, stat])

    def to_dict(self, arrays: bool = False) -> dict:
        d = asdict(self)
        d["percentiles"] = self.percentiles.to_dict()
        if not arrays:
            for k in ("final_equity", "total_return", "max_drawdown", "sharpe"):
                d.pop(k)
        return d


def _path_stats(ret: np.ndarray, periods_per_year: Optional[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ret: (paths x T). Returns (final_equity, max_dd, sharpe) per path."""
    equity = np.cumprod(1.0 + ret, axis=1)
    final = equity[:, -1]
    peak = np.maximum.accumulate(equity, axis=1)
    mdd = (1.0 - equity / peak).max(axis=1)
    mu = ret.mean(axis=1)
    sd = ret.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, np.nan)
    if periods_per_year:
        sr = sr * np.sqrt(periods_per_year)
    return final, mdd, sr


def block_bootstrap(
    returns,
    n_paths: int = 10_000,
    block_length: Optional[float] = None,
    method: str = "stationary",
    periods_per_year: Optional[float] = None,
    seed: Optional[int] = 0,
    percentiles: Sequence[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99),
    chunk_size: int = 1000,
) -> BootstrapResult:
    """Block-bootstrap a return sequence (daily or per-trade) and summarise paths.

    Parameters
    ----------
    returns : 1-D array-like
        Simple returns per period (or per trade). Equity compounds: prod(1 + r).
    n_paths : int
        Number of resampled paths (default 10,000).
    block_length : float, optional
        Mean block length (stationary) or fixed block length (circular).
        Default: ceil(T ** (1/3)), the usual rule of thumb (minimum 2).
    method : "stationary" | "circular"
    periods_per_year : float, optional
        Annualises the Sharpe (e.g. 365 for daily crypto; trades/year for trades).
    seed : int or None
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    n_obs = r.size
    if n_obs < 2:
        raise ValueError("need at least 2 observations")
    if block_length is None:
        block_length = max(2, int(np.ceil(n_obs ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)

    finals = np.empty(n_paths)
    mdds = np.empty(n_paths)
    srs = np.empty(n_paths)
    for start in range(0, n_paths, chunk_size):
        c = min(chunk_size, n_paths - start)
        if method == "stationary":
            idx = stationary_bootstrap_indices(n_obs, c, block_length, rng)
        elif method == "circular":
            idx = circular_block_indices(n_obs, c, int(round(block_length)), rng)
        else:
            raise ValueError("method must be 'stationary' or 'circular'")
        f, d, s = _path_stats(r[idx], periods_per_year)
        finals[start:start + c] = f
        mdds[start:start + c] = d
        srs[start:start + c] = s

    obs_f, obs_d, obs_s = _path_stats(r[None, :], periods_per_year)
    qs = list(percentiles)
    table = pd.DataFrame(
        {
            "final_equity": np.percentile(finals, qs),
            "total_return": np.percentile(finals - 1.0, qs),
            "max_drawdown": np.percentile(mdds, qs),
            "sharpe": np.nanpercentile(srs, qs),
        },
        index=pd.Index(qs, name="percentile"),
    )
    return BootstrapResult(
        n_paths=int(n_paths),
        n_obs=int(n_obs),
        method=method,
        block_length=float(block_length),
        final_equity=finals,
        total_return=finals - 1.0,
        max_drawdown=mdds,
        sharpe=srs,
        percentiles=table,
        prob_loss=float(np.mean(finals < 1.0)),
        observed_final_equity=float(obs_f[0]),
        observed_max_drawdown=float(obs_d[0]),
        observed_sharpe=float(obs_s[0]),
    )
