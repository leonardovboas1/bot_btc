"""Performance metrics from an equity curve, a trades record and bar timestamps.

Sharpe/Sortino are annualised from DAILY returns (equity resampled to the last
mark of each UTC calendar day), with 365 trading days per year (crypto trades
every day). ``daily_returns()`` is the function the stats module (DSR, CSCV,
SPA, bootstrap) consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btclab.engine.sim import EXIT_END, TRADE_DTYPE  # noqa: F401

DAYS_PER_YEAR = 365.0
SECONDS_PER_DAY = 86400.0


def to_datetime_index(timestamps) -> pd.DatetimeIndex:
    """Accept UTC ms int64, datetime64, or a DatetimeIndex and return a tz-naive UTC index."""
    if isinstance(timestamps, pd.DatetimeIndex):
        idx = timestamps
    else:
        ts = np.asarray(timestamps)
        if np.issubdtype(ts.dtype, np.datetime64):
            idx = pd.DatetimeIndex(ts)
        else:
            ts = ts.astype(np.int64)
            # heuristics: ns > 1e17, ms > 1e11, else seconds
            if ts.size and ts.max() > 10**17:
                idx = pd.to_datetime(ts, unit="ns")
            elif ts.size and ts.max() > 10**11:
                idx = pd.to_datetime(ts, unit="ms")
            else:
                idx = pd.to_datetime(ts, unit="s")
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def equity_series(equity, timestamps) -> pd.Series:
    return pd.Series(np.asarray(equity, dtype=np.float64), index=to_datetime_index(timestamps))


def daily_equity(equity, timestamps) -> pd.Series:
    """Equity at the last bar of each UTC calendar day (days without bars dropped)."""
    return equity_series(equity, timestamps).resample("1D").last().dropna()


def daily_returns(equity, timestamps, init_equity: float | None = None) -> pd.Series:
    """Simple daily returns of the equity curve, indexed by UTC date.

    The first day's return is measured against ``init_equity`` (default: the
    first bar's equity), so a strategy that moves on day one is not silently
    dropped. Days with zero equity before and after give 0.
    """
    de = daily_equity(equity, timestamps)
    if de.empty:
        return pd.Series(dtype=np.float64)
    base = float(np.asarray(equity)[0]) if init_equity is None else float(init_equity)
    prev = de.shift(1)
    prev.iloc[0] = base
    with np.errstate(divide="ignore", invalid="ignore"):
        r = de / prev - 1.0
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r.name = "daily_return"
    return r


def max_drawdown(equity) -> tuple[float, int, int, int]:
    """Return (max_dd fraction <= 0, peak_idx, trough_idx, duration_bars).

    Duration is the longest peak-to-recovery stretch in bars (not necessarily
    the one containing the deepest trough); if never recovered it runs to the end.
    """
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return 0.0, 0, 0, 0
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, eq / peak - 1.0, 0.0)
    trough = int(np.argmin(dd))
    peak_idx = int(np.argmax(eq[: trough + 1]))
    # longest underwater stretch
    under = dd < 0
    longest = 0
    run = 0
    for u in under:
        if u:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return float(dd[trough]), peak_idx, trough, int(longest)


def _annualised(r: pd.Series, kind: str) -> float:
    if r.size < 2:
        return float("nan")
    mu = float(r.mean())
    if kind == "sharpe":
        sd = float(r.std(ddof=1))
        return float("nan") if sd == 0 else mu / sd * np.sqrt(DAYS_PER_YEAR)
    down = np.minimum(r.to_numpy(), 0.0)
    dd = float(np.sqrt(np.mean(down**2)))
    if dd == 0:
        return float("inf") if mu > 0 else float("nan")
    return mu / dd * np.sqrt(DAYS_PER_YEAR)


def yearly_returns(equity, timestamps, init_equity: float | None = None) -> dict[int, float]:
    """Return per calendar year: year-end equity / previous year-end equity - 1."""
    s = equity_series(equity, timestamps)
    if s.empty:
        return {}
    base = float(s.iloc[0]) if init_equity is None else float(init_equity)
    ye = s.groupby(s.index.year).last()
    out: dict[int, float] = {}
    prev = base
    for year, val in ye.items():
        out[int(year)] = float(val / prev - 1.0) if prev > 0 else float("nan")
        prev = float(val)
    return out


def compute_metrics(equity, trades, timestamps, init_equity: float | None = None) -> dict:
    """All headline metrics as a flat dict (floats, ints, and ``yearly_returns`` dict)."""
    eq = np.asarray(equity, dtype=np.float64)
    tr = np.asarray(trades)
    idx = to_datetime_index(timestamps)
    n = eq.size
    if n == 0:
        raise ValueError("empty equity curve")
    if idx.size != n:
        raise ValueError("timestamps must align with the equity curve")
    base = float(eq[0]) if init_equity is None else float(init_equity)

    total_return = eq[-1] / base - 1.0 if base > 0 else float("nan")
    span_days = (idx[-1] - idx[0]).total_seconds() / SECONDS_PER_DAY
    years = span_days / DAYS_PER_YEAR
    if years > 0 and eq[-1] > 0 and base > 0:
        cagr = (eq[-1] / base) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    dr = daily_returns(eq, idx, init_equity=base)
    sharpe = _annualised(dr, "sharpe")
    sortino = _annualised(dr, "sortino")

    mdd, peak_i, trough_i, dd_bars = max_drawdown(eq)
    dd_days = _longest_underwater_days(eq, idx)
    calmar = cagr / abs(mdd) if mdd < 0 and not np.isnan(cagr) else float("nan")

    n_trades = int(tr.shape[0])
    if n_trades:
        pnl = tr["pnl"].astype(np.float64)
        wins = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        profit_factor = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else float("nan"))
        win_rate = float((pnl > 0).mean())
        expectancy_bp = float(tr["pnl_pct"].mean() * 1e4)
        bars_held = tr["bars"].astype(np.float64)
        avg_bars = float(bars_held.mean())
        dur_sec = (idx[tr["exit_idx"]].asi8 - idx[tr["entry_idx"]].asi8) / 1e9
        avg_hours = float(dur_sec.mean() / 3600.0)
        n_long = int((tr["side"] == 1).sum())
        n_short = int((tr["side"] == -1).sum())
        total_fees = float(tr["fees"].sum())
        total_funding = float(tr["funding"].sum())
    else:
        profit_factor = float("nan")
        win_rate = float("nan")
        expectancy_bp = float("nan")
        avg_bars = float("nan")
        avg_hours = float("nan")
        n_long = n_short = 0
        total_fees = total_funding = 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "max_drawdown_bars": int(dd_bars),
        "max_drawdown_days": float(dd_days),
        "calmar": float(calmar),
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "expectancy_bp": expectancy_bp,
        "avg_trade_bars": avg_bars,
        "avg_trade_hours": avg_hours,
        "exposure": float(np.count_nonzero(_position_from(tr, n)) / n),
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "total_fees": total_fees,
        "total_funding": total_funding,
        "n_days": int(dr.size),
        "years": float(years),
        "yearly_returns": yearly_returns(eq, idx, init_equity=base),
    }


def _longest_underwater_days(eq: np.ndarray, idx: pd.DatetimeIndex) -> float:
    """Longest stretch from a peak bar to recovery (or the end), in days."""
    under = eq < np.maximum.accumulate(eq)
    if not under.any():
        return 0.0
    edges = np.diff(np.concatenate(([0], under.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1) - 1          # the peak bar before the stretch
    ends = np.minimum(np.flatnonzero(edges == -1), eq.size - 1)  # recovery bar or last bar
    secs = (idx[ends].asi8 - idx[np.maximum(starts, 0)].asi8) / 1e9
    return float(secs.max() / SECONDS_PER_DAY)


def _position_from(trades, n: int) -> np.ndarray:
    """Rebuild an in-market mask from the trades record (bars entry..exit inclusive)."""
    mask = np.zeros(n, dtype=np.bool_)
    for e, x in zip(trades["entry_idx"], trades["exit_idx"]):
        mask[int(e): int(x) + 1] = True
    return mask


def compute_metrics_from_result(result, timestamps, init_equity: float | None = None) -> dict:
    """Convenience wrapper for :class:`btclab.engine.sim.SimResult`."""
    return compute_metrics(result.equity, result.trades, timestamps, init_equity=init_equity)
