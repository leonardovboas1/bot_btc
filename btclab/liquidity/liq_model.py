"""Estimated liquidation levels from OI increases x leverage mix; causal flush detector (H5).

Model (a heuristic, not observed data): each bar's OI increase dOI (BTC) * close is new notional,
split `long_share` long / rest short, spread over leverages by `weights`. A L-x long entered at
close c liquidates near c*(1 - 1/L + mmr); a short near c*(1 + 1/L - mmr). State per price bin
decays by `decay` per bar, shrinks proportionally when OI falls (positions closing), and a
level is cleared when a later bar's range trades through it. Heatmap row t uses bars <= t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import LOG_STEP, asof_series, bin_center, bin_index, rolling_z, tf_delta

DEFAULT_WEIGHTS = {10: 0.30, 25: 0.35, 50: 0.25, 100: 0.10}


def oi_on_bars(metrics: pd.DataFrame, perp_index: pd.DatetimeIndex, tf: str = "1h",
               col: str = "sum_open_interest", max_age: str = "12h") -> np.ndarray:
    """OI (BTC) as-of bar close minus the 5 min publication lag (A1.4), backward only."""
    obs = metrics[col].copy()
    obs.index = obs.index + pd.Timedelta("5min")   # usable only after publication
    return asof_series(obs, perp_index + tf_delta(tf), max_age)


def liq_levels(close, high, low, oi, weights=None, mmr=0.004, decay=0.999, long_share=0.5,
               step=LOG_STEP, price_span=(0.5, 1.6)):
    """Returns dict(long, short: T x P float32 notional grids, bin_lo, step, prices)."""
    weights = weights or DEFAULT_WEIGHTS
    close, high, low, oi = (np.asarray(a, float) for a in (close, high, low, oi))
    T = len(close)
    lo_i = int(bin_index(np.nanmin(low) * price_span[0], step))
    hi_i = int(bin_index(np.nanmax(high) * price_span[1], step))
    P = hi_i - lo_i + 1
    L = np.zeros(P); S = np.zeros(P)
    gl = np.zeros((T, P), np.float32); gs = np.zeros((T, P), np.float32)
    ws = np.array(list(weights.values()), float); ws = ws / ws.sum()
    levs = np.array(list(weights.keys()), float)
    prev_oi = np.nan
    for t in range(T):
        L *= decay; S *= decay
        if np.isfinite(oi[t]) and np.isfinite(prev_oi):
            d = oi[t] - prev_oi
            if d < 0 and prev_oi > 0:
                f = 1 + d / prev_oi
                L *= f; S *= f
        # trade-through: long levels at/above the bar low were hit on the way down,
        # short levels at/below the bar high on the way up. Clear before adding new ones.
        il = int(bin_index(low[t], step)) - lo_i
        ih = int(bin_index(high[t], step)) - lo_i
        L[max(il, 0):] = 0.0
        S[: max(ih + 1, 0)] = 0.0
        if np.isfinite(oi[t]):
            if np.isfinite(prev_oi) and oi[t] > prev_oi:
                notional = (oi[t] - prev_oi) * close[t]
                for lv, w in zip(levs, ws):
                    pl = close[t] * (1 - 1 / lv + mmr)
                    ps = close[t] * (1 + 1 / lv - mmr)
                    if pl > 0:
                        i = int(bin_index(pl, step)) - lo_i
                        if 0 <= i < P: L[i] += notional * long_share * w
                    i = int(bin_index(ps, step)) - lo_i
                    if 0 <= i < P: S[i] += notional * (1 - long_share) * w
            prev_oi = oi[t]
        gl[t] = L; gs[t] = S
    return {"long": gl, "short": gs, "bin_lo": lo_i, "step": step,
            "prices": bin_center(np.arange(lo_i, hi_i + 1), step)}


def liq_flush(close, high, low, oi, k: int = 1, w: int = 720, tau: float = 2.0,
              range_mult: float = 1.5, require_down: bool = True) -> np.ndarray:
    """Causal leverage-flush detector (H5): z(dOI_k) < -tau AND bar range >= range_mult x rolling
    mean range (over w bars, incl. current) AND (optionally) r_k < 0. NaN inputs -> False."""
    close, high, low, oi = (np.asarray(a, float) for a in (close, high, low, oi))
    with np.errstate(divide="ignore", invalid="ignore"):
        doi = np.full(len(oi), np.nan)
        doi[k:] = np.log(oi[k:] / oi[:-k])
        r = np.full(len(close), np.nan)
        r[k:] = np.log(close[k:] / close[:-k])
        rng = (high - low) / close
    mp = max(2, int(0.8 * w))
    avg = pd.Series(rng).rolling(w, min_periods=mp).mean().to_numpy()
    z = rolling_z(doi, w)
    out = (z < -tau) & (rng >= range_mult * avg)
    if require_down:
        out &= r < 0
    return np.where(np.isnan(z) | np.isnan(avg), False, out).astype(bool)


def liq_heatmap_json(res: dict, times, max_cells: int = 200_000) -> dict:
    T, P = res["long"].shape
    stride = max(1, int(np.ceil(T * P / max_cells)))
    sl = slice(None, None, stride)
    return {"times": [t.isoformat() for t in times[sl]], "prices": [float(p) for p in res["prices"]],
            "long": np.round(res["long"][sl]).tolist(), "short": np.round(res["short"][sl]).tolist(),
            "unit": "USD notional (estimated)"}
