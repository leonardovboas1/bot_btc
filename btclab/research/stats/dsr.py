"""Probabilistic and Deflated Sharpe Ratio.

References
----------
Bailey, D. H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier".
    Journal of Risk 15(2).  -> PSR, MinTRL.
Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting and Non-Normality". Journal of
    Portfolio Management 40(5).  -> DSR, E[max SR] under N trials.

All Sharpe ratios in this module are *per-period* (same frequency as the input
returns, e.g. daily) unless explicitly annualised. The PSR / DSR formulas are
frequency-invariant as long as SR, skew, kurtosis, T and the trial-Sharpe
variance are measured at the same frequency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329

__all__ = [
    "sharpe_ratio",
    "return_moments",
    "probabilistic_sharpe_ratio",
    "psr_from_returns",
    "expected_max_sharpe",
    "trial_sharpes",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    "DSRResult",
]


def _clean_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def sharpe_ratio(returns, periods_per_year: Optional[float] = None, ddof: int = 1) -> float:
    """Sharpe ratio of a return series (risk-free rate assumed already removed).

    Returns the per-period Sharpe, or the annualised one when ``periods_per_year``
    is given (multiplied by sqrt(periods_per_year)).
    """
    r = _clean_1d(returns)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    if sd == 0.0:
        return float("nan")
    sr = r.mean() / sd
    if periods_per_year:
        sr *= np.sqrt(periods_per_year)
    return float(sr)


def return_moments(returns) -> tuple[int, float, float, float]:
    """(n_obs, per-period SR, skewness, kurtosis) of a return series.

    Kurtosis is the *non-excess* (Pearson) kurtosis: a normal distribution has 3.
    """
    r = _clean_1d(returns)
    n = int(r.size)
    if n < 3:
        return n, float("nan"), float("nan"), float("nan")
    sr = sharpe_ratio(r)
    skew = float(sps.skew(r, bias=True))
    kurt = float(sps.kurtosis(r, bias=True, fisher=False))
    return n, sr, skew, kurt


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """PSR(SR*) = Prob[true SR > SR*] given an estimated SR over ``n_obs`` periods.

    PSR = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    where g3 is the skewness and g4 the (non-excess) kurtosis of the returns.
    ``sr`` and ``sr_benchmark`` must be per-period (not annualised).
    """
    if n_obs is None or n_obs < 2 or not np.isfinite(sr):
        return float("nan")
    radicand = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if radicand <= 0.0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(radicand)
    return float(sps.norm.cdf(z))


def psr_from_returns(returns, sr_benchmark: float = 0.0) -> float:
    """PSR of a return series against a per-period benchmark Sharpe."""
    n, sr, skew, kurt = return_moments(returns)
    return probabilistic_sharpe_ratio(sr, n, sr_benchmark, skew, kurt)


def expected_max_sharpe(n_trials: int, var_trial_sharpes: float) -> float:
    """Expected maximum Sharpe of ``n_trials`` independent trials under the null
    (true SR = 0 for all), given the cross-sectional variance of the trial SRs.

    E[max SR] ~= sqrt(V) * ( (1 - g) * Z^{-1}(1 - 1/N) + g * Z^{-1}(1 - 1/(N e)) )

    with g the Euler-Mascheroni constant (Bailey & López de Prado 2014, eq. 9).
    Units follow ``var_trial_sharpes`` (per-period in, per-period out).
    """
    if n_trials is None or n_trials <= 1 or not np.isfinite(var_trial_sharpes):
        return 0.0
    sd = np.sqrt(max(var_trial_sharpes, 0.0))
    n = float(n_trials)
    term = (1.0 - EULER_GAMMA) * sps.norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * sps.norm.ppf(
        1.0 - 1.0 / (n * np.e)
    )
    return float(sd * term)


def trial_sharpes(trial_returns, ddof: int = 1) -> np.ndarray:
    """Per-period Sharpe of every column of a T x N matrix of trial returns."""
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    mu = np.nanmean(m, axis=0)
    sd = np.nanstd(m, axis=0, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, np.nan)
    return sr


def min_track_record_length(
    sr: float,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
    alpha: float = 0.95,
) -> float:
    """Minimum number of periods for PSR(SR*) to reach confidence ``alpha``.

    MinTRL = 1 + (1 - g3*SR + (g4 - 1)/4 * SR^2) * (Z_alpha / (SR - SR*))^2
    Returns inf when SR <= SR*.
    """
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return float("inf")
    radicand = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if radicand <= 0.0:
        return float("nan")
    z = sps.norm.ppf(alpha)
    return float(1.0 + radicand * (z / (sr - sr_benchmark)) ** 2)


@dataclass
class DSRResult:
    n_obs: int
    sr: float                    # per-period Sharpe of the candidate
    sr_annualised: float         # nan if periods_per_year not given
    skew: float
    kurt: float                  # non-excess
    n_trials: int
    var_trial_sharpes: float     # per-period units
    expected_max_sr: float       # per-period, the deflation benchmark SR*
    psr: float                   # PSR against sr_benchmark (usually 0)
    dsr: float                   # PSR against expected_max_sr
    min_trl: float               # periods needed for PSR(sr_benchmark) >= alpha
    threshold: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def deflated_sharpe_ratio(
    returns,
    trial_sharpe_values: Optional[Sequence[float]] = None,
    *,
    n_trials: Optional[int] = None,
    var_trial_sharpes: Optional[float] = None,
    sr_benchmark: float = 0.0,
    periods_per_year: Optional[float] = None,
    threshold: float = 0.95,
    alpha: float = 0.95,
) -> DSRResult:
    """Deflated Sharpe Ratio of a candidate return series.

    Parameters
    ----------
    returns : 1-D array-like
        Per-period returns of the selected (best) strategy.
    trial_sharpe_values : array-like, optional
        Per-period Sharpe ratios of *all* N trials that were run to find the
        candidate (including the candidate). Their count and sample variance
        drive the expected maximum Sharpe. Alternatively pass ``n_trials`` and
        ``var_trial_sharpes`` explicitly.
    sr_benchmark : float
        Per-period Sharpe benchmark for the plain PSR and MinTRL (default 0).
    periods_per_year : float, optional
        Only used to report ``sr_annualised``.
    threshold : float
        DSR pass level (plan.md: p > 0.95).
    """
    n, sr, skew, kurt = return_moments(returns)

    if trial_sharpe_values is not None:
        ts = _clean_1d(trial_sharpe_values)
        n_trials = int(ts.size) if n_trials is None else int(n_trials)
        if var_trial_sharpes is None:
            var_trial_sharpes = float(ts.var(ddof=1)) if ts.size > 1 else 0.0
    if n_trials is None:
        n_trials = 1
    if var_trial_sharpes is None:
        var_trial_sharpes = 0.0

    sr_star = expected_max_sharpe(n_trials, var_trial_sharpes)
    psr = probabilistic_sharpe_ratio(sr, n, sr_benchmark, skew, kurt)
    dsr = probabilistic_sharpe_ratio(sr, n, sr_star, skew, kurt)
    min_trl = min_track_record_length(sr, sr_benchmark, skew, kurt, alpha)
    sr_ann = sr * np.sqrt(periods_per_year) if periods_per_year else float("nan")

    return DSRResult(
        n_obs=n,
        sr=float(sr),
        sr_annualised=float(sr_ann),
        skew=float(skew),
        kurt=float(kurt),
        n_trials=int(n_trials),
        var_trial_sharpes=float(var_trial_sharpes),
        expected_max_sr=float(sr_star),
        psr=float(psr),
        dsr=float(dsr),
        min_trl=float(min_trl),
        threshold=float(threshold),
        passed=bool(np.isfinite(dsr) and dsr > threshold),
    )
