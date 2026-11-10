"""Small daily multi-asset portfolio backtester (Amendment 7).

Legs are columns. ``target`` row t holds the weights DECIDED at the close of day t (so they may only use
data <= close t); they are held during day t+1 (``lag`` days) and earn day t+1's leg returns. NaN in
``target`` = "no trade on this leg": the (drifted) position is kept. Costs are charged per leg on the
traded notional |w_new - w_drifted| (fee + slippage), funding is paid on perp legs by the held weight
(long pays positive funding, short receives). Weights are fractions of current equity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btclab.engine.costs import BINANCE_PERP_TAKER, SPOT

PERP_COST = BINANCE_PERP_TAKER.fee_rate + BINANCE_PERP_TAKER.slippage_bp / 1e4   # 5 bp + 2 bp per side
SPOT_COST = SPOT.fee_rate + SPOT.slippage_bp / 1e4                                # 10 bp + 2 bp per side


@dataclass
class Result:
    returns: pd.Series      # daily net portfolio return
    weights: pd.DataFrame   # weights actually held each day (after trade, before the day's return)
    turnover: pd.Series     # daily sum |traded weight|
    cost: pd.Series         # daily cost as fraction of equity


def backtest(ret: pd.DataFrame, target: pd.DataFrame, fee, funding: pd.DataFrame | None = None,
             gross_cap: float | None = None, lag: int = 1) -> Result:
    if lag < 1:
        raise ValueError("lag must be >= 1 (causality)")
    idx, cols = ret.index, ret.columns
    target = target.reindex(index=idx, columns=cols)
    R = ret.to_numpy(float)
    avail = np.isfinite(R)
    R0 = np.where(avail, R, 0.0)
    F = np.zeros_like(R0) if funding is None else \
        np.nan_to_num(funding.reindex(index=idx, columns=cols).to_numpy(float))
    fee = np.broadcast_to(np.asarray(fee, float), (R.shape[1],))
    Tg = target.to_numpy(float)
    n, k = R.shape
    W = np.zeros((n, k))
    out = np.zeros(n)
    turn = np.zeros(n)
    cost = np.zeros(n)
    pre = np.zeros(k)
    for t in range(n):
        if t >= lag:
            tg = Tg[t - lag]
            w = np.where(np.isnan(tg), pre, tg)
        else:
            w = pre.copy()
        w = np.where(avail[t], w, 0.0)
        if gross_cap is not None:
            g = np.abs(w).sum()
            if g > gross_cap:
                w = w * (gross_cap / g)
        tr = np.abs(w - pre)
        c = float((fee * tr).sum())
        r = float((w * (R0[t] - F[t])).sum()) - c
        W[t], out[t], turn[t], cost[t] = w, r, tr.sum(), c
        eq = 1.0 + r
        pre = w * (1.0 + R0[t]) / eq if eq > 1e-9 else np.zeros(k)
    return Result(pd.Series(out, idx, name="ret"), pd.DataFrame(W, idx, cols),
                  pd.Series(turn, idx, name="turnover"), pd.Series(cost, idx, name="cost"))


def metrics(r: pd.Series, turnover: pd.Series | None = None, ppy: int = 365) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, calmar=np.nan, turnover=np.nan, pos_q=np.nan, days=n)
    eq = (1 + r).cumprod()
    cagr = float(eq.iloc[-1] ** (ppy / n) - 1) if eq.iloc[-1] > 0 else -1.0
    peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
    mdd = float(np.max(1 - eq.to_numpy() / peak))
    sd = r.std(ddof=1)
    sh = float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")
    q = (1 + r).groupby(r.index.tz_localize(None).to_period("Q")).prod() - 1
    return dict(sharpe=sh, cagr=cagr, mdd=mdd, calmar=cagr / mdd if mdd > 0 else float("nan"),
                turnover=float(turnover.loc[r.index].mean() * ppy) if turnover is not None else float("nan"),
                pos_q=float((q > 0).mean()), days=n)
