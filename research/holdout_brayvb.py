"""One-shot holdout judgment of brayvb_v2 default 4h-HA (preregistration Amendment 6).

Run ONCE. Signals on the full series (continuous state), evaluation only on bars >= holdout cutoff.
Writes research/reports/holdout_brayvb.md and appends the evaluations to the trials table.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btclab.api.presets import get_preset  # noqa: E402
from btclab.data.loaders import get_cutoff, load_funding, load_ohlcv  # noqa: E402
from btclab.engine.metrics import daily_returns  # noqa: E402
from btclab.research.search.trials import TrialStore, spec_hash  # noqa: E402
from btclab.signals.combiner import run_spec  # noqa: E402

sys.path.insert(0, str(ROOT / "research"))
from xasset_brayvb import run_one as wf_run_one  # noqa: E402  (walk-forward OOS, in-sample only)

PRIMARY = "SOLUSDT"
CONTROL = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "NEARUSDT", "FETUSDT"]
TF = "4h"
PPY = 365


def _stats(r: np.ndarray) -> dict:
    r = np.asarray(r, float)
    n = len(r)
    eq = np.cumprod(1 + r)
    tot = float(eq[-1] - 1) if n else float("nan")
    cagr = float(eq[-1] ** (PPY / n) - 1) if n and eq[-1] > 0 else float("nan")
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    mdd = float(np.max(1 - eq / peak)) if n else float("nan")
    sd = r.std(ddof=1) if n > 1 else 0.0
    sh = float(r.mean() / sd * np.sqrt(PPY)) if sd > 0 else float("nan")
    return dict(sharpe=sh, total=tot, cagr=cagr, mdd=mdd, calmar=(cagr / mdd if mdd > 0 else float("nan")), days=n)


def evaluate(sym: str, src: str) -> dict:
    cutoff = get_cutoff()
    df = load_ohlcv("perp", TF, allow_holdout=True, symbol=sym)            # deliberate one-shot unlock
    fund = load_funding(TF, allow_holdout=True, symbol=sym).reindex(df.index).fillna(0.0)
    df = df.assign(funding=fund.to_numpy())
    spec = get_preset("brayvb_v2")
    spec["tf"] = TF
    if src != "real":
        spec["source"] = src
    res, _ = run_spec(spec, df)
    idx = df.index
    cut_local = cutoff.tz_convert(idx.tz) if idx.tz is not None else cutoff.tz_convert("UTC").tz_localize(None)
    n_cut = int(idx.searchsorted(cut_local))
    eq = np.asarray(res.equity, float)
    base = eq[n_cut - 1] if n_cut > 0 else eq[0]
    dr = daily_returns(eq[n_cut:], idx[n_cut:], init_equity=base).to_numpy()
    # buy & hold of the perp over the same window, long pays funding
    close = df["close"].to_numpy(float)
    bar_r = np.r_[close[n_cut] / close[n_cut - 1] - 1, close[n_cut + 1:] / close[n_cut:-1] - 1] - fund.to_numpy()[n_cut:]
    bh_eq = np.cumprod(1 + bar_r)
    bh = daily_returns(bh_eq, idx[n_cut:], init_equity=1.0).to_numpy()
    tr = res.trades
    ent = np.asarray(tr["entry_idx"]) if len(tr) else np.zeros(0, int)
    pnl = np.asarray(tr["pnl"]) if len(tr) else np.zeros(0)
    sel = ent >= n_cut
    open_at_cut = bool(np.any((ent < n_cut) & (np.asarray(tr["exit_idx"]) >= n_cut))) if len(tr) else False
    return dict(sym=sym, src=src, start=str(idx[n_cut]), end=str(idx[-1]), strat=_stats(dr), bh=_stats(bh),
                trades=int(sel.sum()), win=float((pnl[sel] > 0).mean()) if sel.any() else float("nan"),
                open_at_cut=open_at_cut, spec=spec, daily=dr)


def passes(o: dict) -> dict:
    s, b = o["strat"], o["bh"]
    a = s["sharpe"] > 0
    bdd = s["mdd"] <= 0.6 * b["mdd"]
    c = (s["calmar"] > b["calmar"]) if b["cagr"] > 0 else (s["cagr"] > b["cagr"])
    return dict(a=bool(a), b=bool(bdd), c=bool(c), all=bool(a and bdd and c))


def bootstrap_band(sym: str, src: str, draws=2000, block=20, n_days=None, seed=7) -> tuple[float, float]:
    _, _, series = wf_run_one(sym, TF, src)            # walk-forward OOS daily returns (in-sample data only)
    r = series.to_numpy(float)
    rng = np.random.default_rng(seed)
    n = n_days or len(r)
    out = []
    for _ in range(draws):
        starts = rng.integers(0, len(r) - block, size=n // block + 1)
        x = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = x.std(ddof=1)
        out.append(x.mean() / sd * np.sqrt(PPY) if sd > 0 else 0.0)
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def f(x, d=2, pct=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{d}f}"


def main():
    prim = evaluate(PRIMARY, "ha")
    comp = evaluate(PRIMARY, "real")
    ctrl = [evaluate(s, "ha") for s in CONTROL]
    pv = passes(prim)
    cv = [passes(o) for o in ctrl]
    gen_n = sum(v["a"] and v["b"] for v in cv)
    generalises = gen_n >= 4
    lo, hi = bootstrap_band(PRIMARY, "ha", n_days=prim["strat"]["days"])
    inside = lo <= prim["strat"]["sharpe"] <= hi
    if pv["all"] and generalises:
        verdict = "PASS + generalises: first validated candidate (paper-trade confirmation before any live use)"
    elif pv["all"]:
        verdict = "PASS without generalisation: SOL-specific, likely selection luck; paper trade only"
    else:
        verdict = "FAIL: candidate rejected; holdout spent"

    # trial accounting
    store = TrialStore(None)
    try:
        for o in [prim, comp] + ctrl:
            h = hashlib.sha1((spec_hash(o["spec"]) + o["sym"] + "holdout").encode()).hexdigest()
            s = o["strat"]
            store.con.execute(
                "INSERT INTO trials (spec_hash, spec_json, tf, stage, family, tag, median_oos_sharpe) VALUES (?,?,?,?,?,?,?)",
                [h, json.dumps(dict(o["spec"], symbol=o["sym"])), TF, "holdout", f"holdout:{TF}:{o['src']}",
                 o["sym"], s["sharpe"]])
    except Exception as e:  # noqa: BLE001
        print("trial insert note:", e)
    finally:
        store.close()

    def row(o, v=None):
        s, b = o["strat"], o["bh"]
        cols = [o["sym"], o["src"], f(s["sharpe"]), f(s["total"], pct=True), f(s["cagr"], pct=True), f(s["mdd"], pct=True),
                f(s["calmar"]), str(o["trades"]), f(o["win"], pct=True), f(b["sharpe"]), f(b["total"], pct=True),
                f(b["mdd"], pct=True), f(b["calmar"])]
        if v is not None:
            cols += ["✓" if v["a"] else "✗", "✓" if v["b"] else "✗", "✓" if v["c"] else "✗"]
        return "| " + " | ".join(cols) + " |"

    hdr = ("| asset | src | Sharpe | total | CAGR | max DD | Calmar | trades | win | B&H Sharpe | B&H total | B&H max DD | B&H Calmar | (a) | (b) | (c) |\n"
           "|" + "---|" * 16)
    L = [f"**HOLDOUT VERDICT (Amendment 6): {verdict}**", "",
         "# brayvb_v2 default, 4h, Heikin Ashi signals: one-shot holdout judgment", "",
         f"Window: {prim['start']} → {prim['end']} ({prim['strat']['days']} days). Signals on the full series; equity re-based at the cutoff. "
         f"Costs BINANCE_PERP_TAKER + funding; B&H = long perp incl. funding. Position open at cutoff (SOL): {prim['open_at_cut']}.", "",
         "## Primary: SOLUSDT", "", hdr, row(prim, pv), "",
         f"Primary pass (a ∧ b ∧ c): **{'PASS' if pv['all'] else 'FAIL'}**. "
         f"Consistency (reported, not gating): holdout Sharpe {f(prim['strat']['sharpe'])} vs WF-OOS block-bootstrap 5–95% band "
         f"[{f(lo)}, {f(hi)}] → {'inside' if inside else 'outside'}.", "",
         "## Control: same config on 7 other assets", "", hdr] + [row(o, v) for o, v in zip(ctrl, cv)] + ["",
         f"Generalises ((a) and (b) on ≥ 4/7): **{'YES' if generalises else 'NO'}** ({gen_n}/7).", "",
         "## Comparison (not gating): SOLUSDT with real-candle signals", "", hdr, row(comp, passes(comp)), "",
         "Honesty note: ~8 months cannot separate Sharpe 1.0 from 0.5 (SE ≈ 1.2); this run can refute, not prove. "
         "The holdout is now spent; it must not be used for further selection."]
    out = ROOT / "research" / "reports" / "holdout_brayvb.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"... generalises {gen_n}/7; report {out}")


if __name__ == "__main__":
    main()
