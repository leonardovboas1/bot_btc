"""Shared helpers: absolute log-price bins (shift-invariant => causal), as-of joins, rolling z."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "um_futures"
LOG_STEP = 0.0025  # 0.25 % price bins


def tf_delta(tf) -> pd.Timedelta:
    return pd.Timedelta(tf)


def bin_index(price, step: float = LOG_STEP):
    """Absolute bin id: floor(log(p)/log(1+step)). Independent of the data range."""
    return np.floor(np.log(np.asarray(price, float)) / np.log1p(step)).astype(np.int64)


def bin_center(idx, step: float = LOG_STEP):
    return np.exp((np.asarray(idx, float) + 0.5) * np.log1p(step))


def asof_series(obs: pd.Series, bar_close: pd.DatetimeIndex, max_age="12h") -> np.ndarray:
    """Last observation at or before each bar close, NaN if older than max_age. Never interpolates."""
    obs = obs.dropna().sort_index()
    left = pd.DataFrame({"t": bar_close})
    right = pd.DataFrame({"t": obs.index, "v": obs.to_numpy(float), "obs_t": obs.index})
    m = pd.merge_asof(left, right, on="t", direction="backward")
    age = m["t"] - m["obs_t"]
    v = m["v"].to_numpy(float).copy()
    v[(age > pd.Timedelta(max_age)).to_numpy() | m["obs_t"].isna().to_numpy()] = np.nan
    return v


def rolling_z(x, w: int, min_frac: float = 0.8) -> np.ndarray:
    """Causal rolling z-score (window includes current bar), NaN-skipping, min_periods=0.8w."""
    s = pd.Series(np.asarray(x, float))
    mp = max(2, int(np.ceil(min_frac * w)))
    mu = s.rolling(w, min_periods=mp).mean()
    sd = s.rolling(w, min_periods=mp).std(ddof=0)
    z = (s - mu) / sd.where(sd > 0)
    return z.to_numpy()
