"""White's Reality Check and Hansen's Superior Predictive Ability test.

References
----------
White, H. (2000). "A Reality Check for Data Snooping". Econometrica 68(5).
Hansen, P. R. (2005). "A Test for Superior Predictive Ability". Journal of
    Business & Economic Statistics 23(4).
Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap". JASA 89.

Setup
-----
N strategies with per-period returns r[t, n] and a benchmark b[t] (buy-and-hold,
or zero). Performance differentials d[t, n] = r[t, n] - b[t].
H0: max_n E[d_n] <= 0 (no strategy beats the benchmark).

Reality Check statistic   V   = sqrt(T) * max_n d̄_n
SPA statistic             T_S = max(0, max_n sqrt(T) * d̄_n / ω̂_n)

ω̂_n² is the long-run variance of sqrt(T) d̄_n estimated with the stationary-
bootstrap kernel weights κ(T, i) = (T-i)/T (1-q)^i + i/T (1-q)^{T-i}, q = 1/mean_block.

The bootstrap null distribution recentres each strategy at μ̂_n, where for the
*consistent* p-value (Hansen's p_c)
    μ̂^c_n = d̄_n * 1{ sqrt(T) d̄_n / ω̂_n <= -sqrt(2 log log T) }
i.e. strategies that are significantly poor keep their negative mean (and so
do not inflate the null max), all others are centred at zero. The lower
(μ̂^l = min(d̄, 0)) and upper (μ̂^u = 0, == studentised RC) p-values bound p_c.

Vectorisation: the bootstrap mean for every rep is counts @ D / T, where
``counts`` is the (B x T) matrix of how many times each row was drawn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .bootstrap import resample_counts, stationary_bootstrap_indices

__all__ = ["spa_test", "long_run_variance", "SPAResult"]


def long_run_variance(d: np.ndarray, mean_block: float) -> np.ndarray:
    """Hansen (2005) kernel estimate of Var[sqrt(T) d̄] per column of a (T x N) matrix."""
    t_len = d.shape[0]
    q = min(1.0, 1.0 / float(mean_block))
    x = d - d.mean(axis=0, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * t_len)))
    f = np.fft.rfft(x, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:t_len] / t_len   # biased γ̂_i
    i = np.arange(t_len, dtype=float)
    kappa = (t_len - i) / t_len * (1.0 - q) ** i + i / t_len * (1.0 - q) ** (t_len - i)
    kappa[0] = 0.0  # γ̂_0 enters once, unweighted
    omega2 = acov[0] + 2.0 * (kappa[:, None] * acov).sum(axis=0)
    return np.maximum(omega2.real, 1e-300)


@dataclass
class SPAResult:
    n_strategies: int
    n_obs: int
    n_boot: int
    mean_block: float
    rc_statistic: float
    rc_pvalue: float
    spa_statistic: float
    spa_pvalue: float            # consistent (Hansen p_c)
    spa_pvalue_lower: float
    spa_pvalue_upper: float
    best_index: int              # argmax of studentised mean differential
    best_mean_diff: float        # per-period mean of d for the best strategy
    best_t_stat: float           # sqrt(T) d̄ / ω̂ for best_index
    n_poor: int                  # strategies classified as significantly poor
    threshold: float
    passed: bool                 # spa_pvalue < threshold and rc_pvalue < threshold

    def to_dict(self) -> dict:
        return asdict(self)


def spa_test(
    returns,
    benchmark: Union[float, np.ndarray, pd.Series] = 0.0,
    n_boot: int = 2000,
    mean_block: float = 10.0,
    seed: Optional[int] = 0,
    threshold: float = 0.05,
    chunk_size: int = 500,
) -> SPAResult:
    """White's Reality Check and Hansen's SPA test of N strategies vs a benchmark.

    Parameters
    ----------
    returns : (T x N) array / DataFrame, or 1-D (T,) for a single strategy
    benchmark : scalar (e.g. 0.0) or (T,) series such as buy-and-hold returns
    n_boot : stationary-bootstrap replications (default 2,000)
    mean_block : expected block length of the stationary bootstrap
    threshold : pass level for the p-values (plan.md: p < 0.05)
    """
    r = returns.to_numpy(dtype=float) if isinstance(returns, (pd.DataFrame, pd.Series)) else np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    t_len, n = r.shape
    b = np.asarray(benchmark.to_numpy() if isinstance(benchmark, pd.Series) else benchmark, dtype=float)
    if b.ndim == 0:
        b = np.full(t_len, float(b))
    if b.shape[0] != t_len:
        raise ValueError("benchmark length must match returns")
    d = r - b[:, None]
    good = ~np.isnan(d).any(axis=0)
    d = d[:, good]
    col_map = np.flatnonzero(good)
    n = d.shape[1]
    if n == 0 or t_len < 10:
        raise ValueError("need at least one strategy with complete data and >= 10 periods")

    sqrt_t = np.sqrt(t_len)
    d_mean = d.mean(axis=0)
    omega2 = long_run_variance(d, mean_block)
    omega = np.sqrt(omega2)
    t_stats = sqrt_t * d_mean / omega

    rc_stat = float(sqrt_t * d_mean.max())
    spa_stat = float(max(0.0, t_stats.max()))

    a_thresh = np.sqrt(2.0 * np.log(np.log(t_len)))
    poor = t_stats <= -a_thresh
    mu_c = np.where(poor, d_mean, 0.0)
    mu_l = np.minimum(d_mean, 0.0)
    mu_u = np.zeros(n)

    rng = np.random.default_rng(seed)
    rc_boot = np.empty(n_boot)
    spa_c = np.empty(n_boot)
    spa_l = np.empty(n_boot)
    spa_u = np.empty(n_boot)
    for start in range(0, n_boot, chunk_size):
        c = min(chunk_size, n_boot - start)
        idx = stationary_bootstrap_indices(t_len, c, mean_block, rng)
        counts = resample_counts(idx, t_len)
        boot_mean = (counts @ d) / t_len                     # (c, N)
        centred = boot_mean - d_mean[None, :]                # mean zero under bootstrap
        sl = slice(start, start + c)
        rc_boot[sl] = sqrt_t * centred.max(axis=1)
        z = sqrt_t * centred / omega[None, :]
        shift = sqrt_t / omega[None, :]
        spa_c[sl] = np.maximum(0.0, (z + mu_c[None, :] * shift).max(axis=1))
        spa_l[sl] = np.maximum(0.0, (z + mu_l[None, :] * shift).max(axis=1))
        spa_u[sl] = np.maximum(0.0, (z + mu_u[None, :] * shift).max(axis=1))

    rc_p = float(np.mean(rc_boot >= rc_stat))
    p_c = float(np.mean(spa_c >= spa_stat))
    p_l = float(np.mean(spa_l >= spa_stat))
    p_u = float(np.mean(spa_u >= spa_stat))
    best = int(t_stats.argmax())

    return SPAResult(
        n_strategies=int(n),
        n_obs=int(t_len),
        n_boot=int(n_boot),
        mean_block=float(mean_block),
        rc_statistic=rc_stat,
        rc_pvalue=rc_p,
        spa_statistic=spa_stat,
        spa_pvalue=p_c,
        spa_pvalue_lower=p_l,
        spa_pvalue_upper=p_u,
        best_index=int(col_map[best]),
        best_mean_diff=float(d_mean[best]),
        best_t_stat=float(t_stats[best]),
        n_poor=int(poor.sum()),
        threshold=float(threshold),
        passed=bool(p_c < threshold and rc_p < threshold),
    )
