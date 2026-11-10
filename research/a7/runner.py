"""Shared evaluation helpers for the Amendment 7 design / validation runs."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from research.a7.portfolio import metrics
from research.a7.strategies import (Panel, bench_brayvb, bench_btc, bench_ew, configs)

WARMUP_DAYS = 120   # design / V1: first 120 days of the panel are excluded for strategies AND benchmarks


def run_set(panel: Panel, brayvb: pd.Series | None, btc_panel: Panel | None, start=None, end=None,
            skip_c3: bool = False, only_c1_c2: bool = False) -> dict:
    """name -> dict(metrics..., r=returns Series, turn=turnover Series), evaluated on [start, end)."""
    idx = panel.close.index
    s = pd.Timestamp(start) if start is not None else idx[0] + pd.Timedelta(days=WARMUP_DAYS)
    e = pd.Timestamp(end) if end is not None else idx[-1] + pd.Timedelta(days=1)
    out = {}

    def put(name, r, turn):
        r = r[(r.index >= s) & (r.index < e)]
        m = metrics(r, turn)
        m.update(r=r, turn=turn.loc[r.index])
        out[name] = m

    for name, fn in configs().items():
        if only_c1_c2 and name.startswith("C3"):
            continue
        res = fn(panel)
        put(name, res.returns, res.turnover)
    ew = bench_ew(panel)
    put("EW_BH", ew.returns, ew.turnover)
    bp = btc_panel if btc_panel is not None else panel
    b = bench_btc(bp)
    put("BTC_BH", b.returns.reindex(idx).fillna(0.0), b.turnover.reindex(idx).fillna(0.0))
    if brayvb is not None:
        r = brayvb.reindex(idx).fillna(0.0)
        put("BRAYVB", r, pd.Series(np.nan, idx))
    return out


def fmt(x, pct=False, d=2):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{d}f}"


def table(res: dict) -> str:
    L = ["| name | Sharpe | CAGR | max DD | Calmar | turnover/yr | % pos quarters | days |", "|" + "---|" * 8]
    for n, m in res.items():
        L.append(f"| {n} | {fmt(m['sharpe'])} | {fmt(m['cagr'], True)} | {fmt(m['mdd'], True)} | "
                 f"{fmt(m['calmar'])} | {fmt(m['turnover'], d=1)} | {fmt(m['pos_q'], True)} | {m['days']} |")
    return "\n".join(L)


def verdict(res: dict, name: str, only_c3_rules=None) -> dict:
    """A7 success rules on one set. C1/C2: needs EW_BH and BRAYVB; C3: absolute."""
    m = res[name]
    if name.startswith("C3"):
        a, b, c = m["sharpe"] >= 1.0, m["mdd"] <= 0.15, m["pos_q"] >= 0.70
        return dict(ok=bool(a and b and c), detail=f"Sharpe {fmt(m['sharpe'])}>=1.0:{a}; DD {fmt(m['mdd'], True)}<=15%:{b}; "
                                                    f"posQ {fmt(m['pos_q'], True)}>=70%:{c}")
    ew, br = res["EW_BH"], res.get("BRAYVB")
    a = m["sharpe"] >= ew["sharpe"] + 0.3
    b = m["mdd"] <= 0.7 * ew["mdd"]
    c = bool(br is not None and np.isfinite(br["sharpe"]) and m["sharpe"] >= br["sharpe"])
    return dict(ok=bool(a and b and c),
                detail=f"Sharpe {fmt(m['sharpe'])}>=EW {fmt(ew['sharpe'])}+0.3:{a}; DD {fmt(m['mdd'], True)}<=0.7*{fmt(ew['mdd'], True)}:{b}; "
                       f"Sharpe>=brayvb {fmt(br['sharpe']) if br else 'n/a'}:{c}")
