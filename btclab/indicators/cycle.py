"""Cycle family: Mayer multiple, Pi Cycle Top. Designed for daily bars; ``bars_per_day``
scales the day-based lengths for intraday base timeframes (e.g. 6 for 4h)."""
from __future__ import annotations

import numpy as np

from .base import crossover, crossunder, f64, sma


def mayer(close, length: int = 200, hot: float = 2.4, cold: float = 0.8, bars_per_day: int = 1) -> dict:
    c = f64(close)
    m = c / sma(c, length * bars_per_day)
    return {"mayer": m, "overheated": m > hot, "undervalued": m < cold, "above_1": m > 1.0,
            "cross_above_hot": crossover(m, hot), "cross_below_cold": crossunder(m, cold)}


def pi_cycle(close, fast: int = 111, slow: int = 350, bars_per_day: int = 1) -> dict:
    c = f64(close)
    f = sma(c, fast * bars_per_day)
    s = 2.0 * sma(c, slow * bars_per_day)
    return {"fast_ma": f, "slow_ma2": s, "top": crossover(f, s), "bottom": crossunder(f, s),
            "fast_above_slow": f > s}
