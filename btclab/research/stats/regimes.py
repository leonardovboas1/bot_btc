"""Causal regime labelling and per-regime performance breakdown.

Regime at day t = sign of the 200-day SMA slope  x  realised-volatility tercile.

Everything is computed from data up to and including t:
* slope sign : sign(SMA_t - SMA_{t-slope_lag})
* realised vol : rolling std of log returns over ``vol_window`` days
* terciles : the vol series is bucketed against *expanding-window* quantiles
  (33 % / 67 % of all vol values observed up to t), never full-sample
  quantiles, so labels never change when future data is appended.

Labels: "{up|down}_{low|mid|high}vol"; NaN during warm-up.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = ["label_regimes", "regime_performance", "REGIME_LABELS"]

TREND = {1: "up", -1: "down"}
VOL = {0: "lowvol", 1: "midvol", 2: "highvol"}
REGIME_LABELS = [f"{t}_{v}" for t in ("up", "down") for v in ("lowvol", "midvol", "highvol")]


def label_regimes(
    close: pd.Series,
    sma_window: int = 200,
    slope_lag: int = 1,
    vol_window: int = 20,
    min_vol_history: Optional[int] = None,
) -> pd.Series:
    """Causal regime label per bar of a daily close series.

    Parameters
    ----------
    close : pd.Series (DatetimeIndex, daily)
    sma_window : SMA length (200)
    slope_lag : bars over which the SMA slope sign is measured (1 = one-day change)
    vol_window : realised-vol lookback (rolling std of log returns)
    min_vol_history : vol observations required before terciles are defined
        (default: ``vol_window``). Earlier bars are NaN.
    """
    c = pd.Series(close).astype(float)
    sma = c.rolling(sma_window, min_periods=sma_window).mean()
    slope = sma - sma.shift(slope_lag)
    trend = np.sign(slope)
    trend = trend.where(trend != 0, np.nan)  # flat slope: undefined (rare)
    # a zero slope is extremely rare; treat it as a continuation of the prior sign
    trend = trend.ffill()

    logret = np.log(c).diff()
    vol = logret.rolling(vol_window, min_periods=vol_window).std()
    min_hist = vol_window if min_vol_history is None else min_vol_history
    q33 = vol.expanding(min_periods=min_hist).quantile(1.0 / 3.0)
    q67 = vol.expanding(min_periods=min_hist).quantile(2.0 / 3.0)
    bucket = pd.Series(np.where(vol > q67, 2, np.where(vol > q33, 1, 0)), index=c.index, dtype=float)
    bucket[vol.isna() | q33.isna() | q67.isna()] = np.nan

    valid = trend.notna() & bucket.notna() & sma.notna()
    labels = pd.Series(np.nan, index=c.index, dtype=object)
    labels[valid] = [
        f"{TREND[int(t)]}_{VOL[int(b)]}" for t, b in zip(trend[valid], bucket[valid])
    ]
    labels.name = "regime"
    return labels


def _stats(r: np.ndarray, periods_per_year: Optional[float]) -> dict:
    n = r.size
    if n == 0:
        return {"n": 0, "mean": np.nan, "total_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    mu = r.mean()
    sd = r.std(ddof=1) if n > 1 else np.nan
    sr = mu / sd if sd and sd > 0 else np.nan
    if periods_per_year and np.isfinite(sr):
        sr *= np.sqrt(periods_per_year)
    eq = np.cumprod(1.0 + r)
    mdd = float((1.0 - eq / np.maximum.accumulate(eq)).max())
    return {"n": int(n), "mean": float(mu), "total_return": float(eq[-1] - 1.0), "sharpe": float(sr), "max_drawdown": mdd}


def regime_performance(
    returns: pd.Series,
    regimes: pd.Series,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: Optional[float] = None,
    compare_on: str = "mean",
) -> pd.DataFrame:
    """Per-regime statistics of a strategy (and optionally a benchmark).

    Rows: one per regime label present (+ "all"). Columns: n, mean,
    total_return, sharpe, max_drawdown and, when ``benchmark`` is given, the
    same with a ``bench_`` prefix plus ``excess`` (strategy - benchmark on
    ``compare_on``) and ``worse_than_benchmark``.
    """
    r = pd.Series(returns).astype(float)
    reg = pd.Series(regimes).reindex(r.index)
    b = pd.Series(benchmark).astype(float).reindex(r.index) if benchmark is not None else None
    rows = {}
    groups = [("all", reg.notna())] + [(lab, reg == lab) for lab in REGIME_LABELS if (reg == lab).any()]
    for lab, mask in groups:
        m = mask & r.notna()
        if b is not None:
            m &= b.notna()
        d = _stats(r[m].to_numpy(), periods_per_year)
        if b is not None:
            bd = _stats(b[m].to_numpy(), periods_per_year)
            d.update({f"bench_{k}": v for k, v in bd.items() if k != "n"})
            d["excess"] = d[compare_on] - bd[compare_on]
            d["worse_than_benchmark"] = bool(d["excess"] < 0)
        rows[lab] = d
    out = pd.DataFrame(rows).T
    out.index.name = "regime"
    out["n"] = out["n"].astype(int)
    return out
