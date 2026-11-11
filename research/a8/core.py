"""Shared evaluation helpers for Amendment 8 (design + validation). Threads pinned to 1."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from btclab.engine.metrics import daily_returns  # noqa: E402
from btclab.engine.sim import simulate  # noqa: E402
from btclab.indicators import base as _base  # noqa: E402
from btclab.signals.combiner import (ATR_LEN, _cached_compute, _cost_model, heikin_ashi)  # noqa: E402
from btclab.signals.dsl import RISK_KEYS, build_signal, validate_spec  # noqa: E402
from btclab.research.search.core import PPY, _max_dd, _sh, summarize  # noqa: E402

DESIGN = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "NEARUSDT", "FETUSDT"]


def run_fast(spec: dict, df: pd.DataFrame, cache: dict):
    """run_spec without the metrics pass (same semantics)."""
    validate_spec(spec)
    n = len(df)
    inds = spec["indicators"]
    source = spec.get("source", "real")
    sig_df = df
    if source == "ha":
        hk = ("__ha__", id(df), len(df))
        if hk not in cache:
            cache[hk] = heikin_ashi(df)
        sig_df = cache[hk]

    def lookup(prim):
        name, _, key = prim.partition(".")
        return _cached_compute(cache, sig_df, name, inds[name], source)[key]

    sigs = {s: build_signal(spec.get(s), lookup, n) for s in ("long", "short", "exit_long", "exit_short")}
    risk = {k: v for k, v in (spec.get("risk") or {}).items() if k in RISK_KEYS and v}
    kw = _cost_model(spec.get("costs", "BINANCE_PERP_TAKER")).sim_kwargs()
    if sigs["short"] is None:
        kw["allow_short"] = False

    def col(c):
        return df[c].to_numpy(dtype=np.float64)

    atr = _base.atr(col("high"), col("low"), col("close"), ATR_LEN)
    funding = col("funding") if "funding" in df.columns else None
    return simulate(col("open"), col("high"), col("low"), col("close"), sigs["long"], sigs["short"],
                    sigs["exit_long"], sigs["exit_short"], atr=atr, funding=funding, **kw, **risk)


_CTX: dict = {}
_CACHE: dict = {}


def get_ctx(sym, tf):
    k = (sym, tf)
    if k not in _CTX:
        from xasset_brayvb import make_ctx
        _CTX[k] = make_ctx(sym, tf)[0]
    return _CTX[k]


def coin_cache(sym, tf):
    c = _CACHE.setdefault((sym, tf), {})
    if len(c) > 30:
        c.clear()
    return c


def eval_coin(spec, sym, tf):
    ctx = get_ctx(sym, tf)
    res = run_fast(spec, ctx.df, coin_cache(sym, tf))
    rec = summarize(ctx, res)
    tr = res.trades
    ent = np.asarray(tr["entry_idx"])
    sel = ctx.oos_bar_mask[ent] if len(ent) else np.zeros(0, bool)
    pnl = np.asarray(tr["pnl"])[sel]
    side = np.asarray(tr["side"])[sel]
    dr = daily_returns(res.equity, ctx.df.index).reindex(ctx.days).fillna(0.0).to_numpy()
    oos = pd.Series(dr[ctx.oos_mask], index=ctx.days[ctx.oos_mask])
    return dict(sym=sym, is_sharpe=rec["mean_is_sharpe"], oos_sharpe=rec["oos_sharpe"], n_folds=len(ctx.folds),
                oos_trades=int(sel.sum()), is_trades=rec["is_trades"], pnl=pnl, side=side, oos=oos,
                max_dd=rec["max_dd"])


def pooled_pf(pnl):
    w, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (99.0 if w > 0 else 0.0)


def side_gate(pnl, side):
    """Short leg pooled PnL >= 0, or its loss <= 20% of the long leg's gross loss contribution."""
    sp = pnl[side < 0].sum()
    if sp >= 0:
        return True
    long_loss = -pnl[(side > 0) & (pnl < 0)].sum()
    return bool(-sp <= 0.2 * long_loss)


def aggregate(coins: list) -> dict:
    """coins = list of eval_coin dicts (coins with n_folds == 0 are ignored)."""
    coins = [c for c in coins if c["n_folds"] > 0]
    pnl = np.concatenate([c["pnl"] for c in coins]) if coins else np.zeros(0)
    side = np.concatenate([c["side"] for c in coins]) if coins else np.zeros(0)
    osh = np.array([c["oos_sharpe"] for c in coins])
    pool = pd.concat([c["oos"] for c in coins], axis=1).mean(axis=1) if coins else pd.Series(dtype=float)
    return dict(
        obj=float(np.median([c["is_sharpe"] for c in coins])) if coins else -10.0,
        score=float(np.median(osh)) if coins else -10.0,
        n_pos=int((osh > 0).sum()), n_coins=len(coins),
        pf=pooled_pf(pnl), side_ok=side_gate(pnl, side),
        short_pnl=float(pnl[side < 0].sum()), long_pnl=float(pnl[side > 0].sum()),
        min_trades=int(min(c["oos_trades"] for c in coins)) if coins else 0,
        is_trades=int(sum(c["is_trades"] for c in coins)),
        pool=pool, pool_sharpe=_sh(pool.to_numpy()), pool_dd=_max_dd(pool.to_numpy()),
        per_coin={c["sym"]: round(float(c["oos_sharpe"]), 3) for c in coins},
    )


def gates(a: dict, min_trades=30) -> dict:
    g = dict(pos=a["n_pos"] >= 6, pf=a["pf"] > 1.1, side=a["side_ok"], trades=a["min_trades"] >= min_trades)
    g["all"] = all(g.values())
    return g
