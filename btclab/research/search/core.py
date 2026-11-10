"""Evaluation core: walk-forward context, per-spec fold metrics, parallel batch evaluation."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from btclab.engine.metrics import daily_returns
from btclab.research.stats.dsr import sharpe_ratio
from btclab.research.stats.walkforward import walk_forward_folds
from btclab.signals.combiner import run_spec

PPY = 365.0

# Plan C constraints (>=100 IS trades, >=30 per OOS fold). Daily bars cannot produce 30 trades in a
# 6-month fold for slow primitives, so 1d uses scaled-down minimums (documented in every report).
CONSTRAINTS = {
    "1h": dict(min_is_trades=100, min_oos_trades=30),
    "4h": dict(min_is_trades=100, min_oos_trades=30),
    "1d": dict(min_is_trades=40, min_oos_trades=8),
}
TIME_STOP = {"1h": 24, "4h": 12, "1d": 5}
SL_ATR, TP_ATR = 2.0, 4.0
PURGE_BARS = {"1h": 200, "4h": 200, "1d": 200}


@dataclass
class Context:
    tf: str
    df: pd.DataFrame
    folds: list
    days: pd.DatetimeIndex
    test_masks: List[np.ndarray]
    train_masks: List[np.ndarray]
    test_bars: List[tuple]
    oos_mask: np.ndarray
    train_bar_mask: np.ndarray
    oos_bar_mask: np.ndarray
    bench: np.ndarray                 # buy&hold daily returns on days
    close_daily: pd.Series
    constraints: dict = field(default_factory=dict)
    loader: bool = True               # True: workers can rebuild it from the data loaders

    @property
    def oos_days(self) -> pd.DatetimeIndex:
        return self.days[self.oos_mask]


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def build_context(tf: str, df: Optional[pd.DataFrame] = None, cutoff=None, n_folds: int = 6,
                  constraints: Optional[dict] = None) -> Context:
    loader = df is None
    if df is None:
        from btclab.data.loaders import get_cutoff, load_funding, load_ohlcv
        cutoff = get_cutoff()                       # NEVER past the holdout cutoff
        df = load_ohlcv("perp", tf, end=cutoff)
        df = df.assign(funding=load_funding(tf, end=cutoff).reindex(df.index).fillna(0.0).to_numpy())
    idx = df.index
    folds = walk_forward_folds(idx, n_folds=n_folds, purge_bars=PURGE_BARS.get(tf, 200),
                               embargo="5D", end=cutoff)
    naive = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
    days = pd.DatetimeIndex(naive.normalize().unique())
    test_masks, train_masks, test_bars = [], [], []
    train_bar = np.zeros(len(df), bool)
    oos_bar = np.zeros(len(df), bool)
    for f in folds:
        test_masks.append((days >= _naive(f.test_start).normalize()) & (days <= _naive(f.test_end).normalize()))
        train_masks.append((days >= _naive(f.train_start).normalize()) & (days <= _naive(f.train_end).normalize()))
        test_bars.append((int(f.test_idx[0]), int(f.test_idx[-1])))
        train_bar[f.train_idx] = True
        oos_bar[f.test_idx] = True
    oos_mask = np.any(test_masks, axis=0)
    close_d = pd.Series(df["close"].to_numpy(), index=naive)
    close_d = close_d.groupby(close_d.index.normalize()).last().reindex(days)
    bench = close_d.pct_change().fillna(0.0).to_numpy()
    cons = {**CONSTRAINTS.get(tf, CONSTRAINTS["1d"]), **(constraints or {})}
    return Context(tf, df, folds, days, test_masks, train_masks, test_bars, oos_mask, train_bar, oos_bar,
                   bench, close_d, cons, loader)


def _sh(x: np.ndarray) -> float:
    s = sharpe_ratio(x, PPY) if len(x) > 2 else 0.0
    return 0.0 if not np.isfinite(s) else float(s)


def _max_dd(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    return float((1.0 - eq / np.maximum.accumulate(eq)).max())


def summarize(ctx: Context, res) -> dict:
    """Fold metrics and OOS series of one simulation result (full-history run sliced into folds)."""
    dr = (daily_returns(res.equity, ctx.df.index).reindex(ctx.days).fillna(0.0)).to_numpy()
    tr = res.trades
    entry = tr["entry_idx"]
    folds = []
    for k, (tm, trm, (lo, hi)) in enumerate(zip(ctx.test_masks, ctx.train_masks, ctx.test_bars)):
        n_oos = int(((entry >= lo) & (entry <= hi)).sum())
        folds.append(dict(fold=k, oos_sharpe=_sh(dr[tm]), oos_ret=float(np.prod(1 + dr[tm]) - 1),
                          oos_trades=n_oos, is_sharpe=_sh(dr[trm])))
    oos = dr[ctx.oos_mask]
    in_oos = ctx.oos_bar_mask[entry] if len(entry) else np.zeros(0, bool)
    pnl = tr["pnl"][in_oos]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = float(wins / losses) if losses > 0 else (99.0 if wins > 0 else 0.0)
    sh = [f["oos_sharpe"] for f in folds]
    return dict(
        fold_json=json.dumps(folds),
        median_oos_sharpe=float(np.median(sh)),
        mean_is_sharpe=float(np.mean([f["is_sharpe"] for f in folds])),
        n_pos_folds=int(sum(s > 0 for s in sh)),
        is_trades=int(ctx.train_bar_mask[entry].sum()) if len(entry) else 0,
        min_oos_trades=int(min(f["oos_trades"] for f in folds)),
        oos_sharpe=_sh(oos), max_dd=_max_dd(oos), profit_factor=min(pf, 99.0),
        oos_returns=oos.astype(np.float32).tobytes(),
    )


def evaluate_spec(ctx: Context, spec: dict, cache: Optional[dict] = None) -> dict:
    res, _ = run_spec(spec, ctx.df, cache)
    return summarize(ctx, res)


def eligible(row, c: dict) -> bool:
    """Plan-C trade-count constraints plus a positive median OOS Sharpe."""
    return bool(row["is_trades"] >= c["min_is_trades"] and row["min_oos_trades"] >= c["min_oos_trades"]
                and row["median_oos_sharpe"] > 0)


def top_fraction(rows: pd.DataFrame, c: dict, frac: float = 0.15) -> List[str]:
    """Hashes of the top frac of eligible rows by median OOS Sharpe."""
    if rows.empty:
        return []
    ok = rows[rows.apply(lambda r: eligible(r, c), axis=1)]
    if ok.empty:
        return []
    k = max(1, math.ceil(frac * len(ok)))
    return ok.sort_values("median_oos_sharpe", ascending=False).head(k)["spec_hash"].tolist()


# ------------------------------------------------------------------ parallel evaluation
_CTX: Dict[str, Context] = {}


def get_ctx(tf: str, kw: Optional[dict] = None) -> Context:
    if tf not in _CTX:
        _CTX[tf] = build_context(tf, **(kw or {}))
    return _CTX[tf]


def _eval_batch(tf: str, specs: List[dict], kw: dict, meta: List[dict]) -> List[dict]:
    from btclab.research.search.trials import spec_hash
    ctx = get_ctx(tf, kw)
    cache: dict = {}                                   # indicator cache shared across the batch
    out = []
    for s, m in zip(specs, meta):
        h = spec_hash(s)
        try:
            rec = evaluate_spec(ctx, s, cache)
        except Exception as e:                         # a failing config is not a trial
            out.append(dict(spec_hash=h, error=repr(e)))
            continue
        out.append(dict(rec, spec_json=json.dumps(s), spec_hash=h, tf=tf, **m))
    return out


def n_jobs_default() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def evaluate_many(ctx: Optional[Context], tf: str, specs: List[dict], meta: List[dict],
                  n_jobs: int = 1, batch: int = 30, ctx_kw: Optional[dict] = None, log=print):
    """Generator of record lists. ctx is used directly when n_jobs == 1 or it is not loader-built."""
    ctx_kw = ctx_kw or {}
    order = sorted(range(len(specs)), key=lambda i: sorted(specs[i]["indicators"]))
    specs = [specs[i] for i in order]
    meta = [meta[i] for i in order]
    chunks = [(specs[i:i + batch], meta[i:i + batch]) for i in range(0, len(specs), batch)]
    if n_jobs == 1 or (ctx is not None and not ctx.loader):
        if ctx is not None:
            _CTX[tf] = ctx
        for s, m in chunks:
            yield _eval_batch(tf, s, ctx_kw, m)
        return
    from joblib import Parallel, delayed
    gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(_eval_batch)(tf, s, ctx_kw, m) for s, m in chunks)
    for i, recs in enumerate(gen, 1):
        log(f"    batch {i}/{len(chunks)}")
        yield recs
