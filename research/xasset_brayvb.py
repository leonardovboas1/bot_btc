"""Cross-asset robustness of the brayvb_v2 DEFAULT preset (docs/preregistration.md Amendment 5). ZERO tuning.

Run:  python research/xasset_brayvb.py      (single process; holdout never loaded)
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btclab.api.presets import get_preset
from btclab.data.loaders import get_cutoff, load_funding, load_ohlcv
from btclab.engine.metrics import daily_returns
from btclab.research.search.core import PPY, PURGE_BARS, CONSTRAINTS, Context, _max_dd, _naive, _sh, summarize
from btclab.research.search.trials import TrialStore, spec_hash
from btclab.research.stats import deflated_sharpe_ratio
from btclab.research.stats.walkforward import Fold
from btclab.signals.combiner import run_spec

ALTS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "NEARUSDT", "FETUSDT"]
ASSETS = ["BTCUSDT"] + ALTS
TFS, SOURCES = ("1d", "4h"), ("real", "ha")
MIN_TRAIN_MONTHS = 12.0
REPORT = ROOT / "research" / "reports" / "xasset_brayvb.md"


def build_folds(idx: pd.DatetimeIndex, tf: str, cutoff, n_folds=6, train_m=18, test_m=6):
    """Same calendar test windows for every asset; train truncated at first bar; <12m train -> dropped."""
    emb = pd.Timedelta("5D")
    first = idx[0]
    kept, dropped = [], []
    for k in range(n_folds):
        back = n_folds - 1 - k
        t_end = cutoff - pd.DateOffset(months=test_m * back)
        t_start = cutoff - pd.DateOffset(months=test_m * back + test_m)
        tr_end = t_start - emb
        tr_start = tr_end - pd.DateOffset(months=train_m)
        avail = (tr_end - max(tr_start, first)).days / 30.4375
        if avail < MIN_TRAIN_MONTHS:
            dropped.append((k, round(max(avail, 0.0), 1)))
            continue
        trp = np.flatnonzero((idx >= tr_start) & (idx < tr_end))
        pg = PURGE_BARS.get(tf, 200)
        trp = trp[:-pg] if pg < trp.size else trp[:0]
        tep = np.flatnonzero((idx >= t_start) & (idx < t_end))
        kept.append(Fold(k, trp, tep, idx[trp[0]], idx[trp[-1]], idx[tep[0]], idx[tep[-1]], pg, emb))
    return kept, dropped


def make_ctx(sym: str, tf: str):
    cutoff = get_cutoff()
    df = load_ohlcv("perp", tf, end=cutoff, symbol=sym)                       # holdout enforced
    df = df.assign(funding=load_funding(tf, end=cutoff, symbol=sym).reindex(df.index).fillna(0.0).to_numpy())
    idx = df.index
    folds, dropped = build_folds(idx, tf, cutoff)
    naive = idx.tz_convert("UTC").tz_localize(None)
    days = pd.DatetimeIndex(naive.normalize().unique())
    tms, trms, tbars = [], [], []
    train_bar, oos_bar = np.zeros(len(df), bool), np.zeros(len(df), bool)
    for f in folds:
        tms.append((days >= _naive(f.test_start).normalize()) & (days <= _naive(f.test_end).normalize()))
        trms.append((days >= _naive(f.train_start).normalize()) & (days <= _naive(f.train_end).normalize()))
        tbars.append((int(f.test_idx[0]), int(f.test_idx[-1])))
        train_bar[f.train_idx] = True
        oos_bar[f.test_idx] = True
    close_d = pd.Series(df["close"].to_numpy(), index=naive)
    close_d = close_d.groupby(close_d.index.normalize()).last().reindex(days)
    bench = close_d.pct_change().fillna(0.0).to_numpy()
    ctx = Context(tf, df, folds, days, tms, trms, tbars, np.any(tms, axis=0), train_bar, oos_bar, bench,
                  close_d, dict(CONSTRAINTS[tf]), False)
    return ctx, dropped


def stats(r: np.ndarray) -> dict:
    n = len(r)
    eq = float(np.prod(1 + r))
    cagr = eq ** (PPY / n) - 1 if n and eq > 0 else float("nan")
    dd = _max_dd(r)
    return dict(sharpe=_sh(r), cagr=cagr, mdd=dd, calmar=(cagr / dd if dd > 0 else float("nan")))


def run_one(sym, tf, src):
    ctx, dropped = make_ctx(sym, tf)
    spec = get_preset("brayvb_v2")                   # DEFAULT preset, untouched
    spec["tf"] = tf
    if src != "real":
        spec["source"] = src
    res, _ = run_spec(spec, ctx.df)
    rec = summarize(ctx, res)
    dr = daily_returns(res.equity, ctx.df.index).reindex(ctx.days).fillna(0.0).to_numpy()
    m = ctx.oos_mask
    entry, pnl = np.asarray(res.trades["entry_idx"]), np.asarray(res.trades["pnl"])
    sel = ctx.oos_bar_mask[entry] if len(entry) else np.zeros(0, bool)
    p = pnl[sel]
    out = dict(sym=sym, tf=tf, src=src, folds_used=len(ctx.folds), dropped=dropped, n_days=int(m.sum()),
               strat=stats(dr[m]), bh=stats(ctx.bench[m]), trades=int(sel.sum()),
               win=float((p > 0).mean()) if len(p) else float("nan"))
    h = hashlib.sha1((spec_hash(spec) + sym).encode()).hexdigest()     # spec_hash ignores symbol -> salt it
    row = dict(rec, spec_json=json.dumps(dict(spec, symbol=sym)), spec_hash=h, tf=tf, stage="xasset",
               parent_hash=None, family=f"xasset:{tf}:{src}", tag=sym)
    series = pd.Series(dr[m], index=ctx.days[m])
    return out, row, series


def f2(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def pc(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def main():
    results, rows, series = {}, [], {}
    for tf in TFS:
        for src in SOURCES:
            for sym in ASSETS:
                o, row, s = run_one(sym, tf, src)
                results[(tf, src, sym)] = o
                series[(tf, src, sym)] = s
                rows.append(row)
                print(tf, src, sym, o["folds_used"], f2(o["strat"]["sharpe"]), flush=True)

    store = TrialStore()
    n_before = int(store.con.execute("SELECT count(*) FROM trials WHERE stage<>'xasset'").fetchone()[0])
    store.insert_many(rows)
    n_x = int(store.con.execute("SELECT count(*) FROM trials WHERE stage='xasset'").fetchone()[0])
    N = n_before + 32 + 1
    store.close()

    alt_sr = []
    for t in TFS:
        for sr in SOURCES:
            for s in ALTS:
                v = series[(t, sr, s)].to_numpy()
                if np.std(v, ddof=1) > 0:
                    alt_sr.append(np.mean(v) / np.std(v, ddof=1))
    alt_sr = np.array(alt_sr)

    verdicts, tables = [], []
    for tf in TFS:
        for src in SOURCES:
            cfg = f"{tf}:{src}"
            a = [results[(tf, src, s)] for s in ALTS]
            sh = [x["strat"]["sharpe"] for x in a]
            med = float(np.median(sh))
            n_pos = sum(s > 0 for s in sh)
            n_cal = sum(bool(np.isfinite(x["strat"]["calmar"]) and np.isfinite(x["bh"]["calmar"])
                             and x["strat"]["calmar"] > x["bh"]["calmar"]) for x in a)
            n_dd = sum(bool(x["strat"]["mdd"] <= 0.6 * x["bh"]["mdd"]) for x in a)
            robust = med > 0 and n_pos >= 5 and n_cal >= 4
            less = n_dd >= 5
            pool = pd.concat([series[(tf, src, s)] for s in ALTS], axis=1).mean(axis=1, skipna=True)
            d = deflated_sharpe_ratio(pool.to_numpy(), alt_sr, n_trials=N, periods_per_year=PPY)
            ps = stats(pool.to_numpy())
            verdicts.append(
                f"- `{cfg}`: robust across assets **{'PASS' if robust else 'FAIL'}** "
                f"(median alt Sharpe {f2(med)}; Sharpe>0 on {n_pos}/7; Calmar beats B&H on {n_cal}/7) | "
                f"loses-less (maxDD <= 0.6 x B&H on >=5/7) **{'PASS' if less else 'FAIL'}** ({n_dd}/7) | "
                f"pooled-alts Sharpe {f2(ps['sharpe'])}, DSR p = {f2(d.dsr, 3)} (N = {N})")
            t = [f"### Config {cfg}", "",
                 "| asset | OOS Sharpe | CAGR | max DD | Calmar | trades | win % | B&H Sharpe | B&H CAGR | B&H max DD | B&H Calmar | folds used |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for s in ASSETS:
                x = results[(tf, src, s)]
                st, bh = x["strat"], x["bh"]
                drop = f" (dropped {[k for k, _ in x['dropped']]})" if x["dropped"] else ""
                t.append(f"| {s}{' (ref)' if s == 'BTCUSDT' else ''} | {f2(st['sharpe'])} | {pc(st['cagr'])} | {pc(st['mdd'])} | "
                         f"{f2(st['calmar'])} | {x['trades']} | {pc(x['win'])} | {f2(bh['sharpe'])} | {pc(bh['cagr'])} | "
                         f"{pc(bh['mdd'])} | {f2(bh['calmar'])} | {x['folds_used']}/6{drop} |")
            tables.append("\n".join(t))
    dropped_notes = [(s, [k for k, _ in results[("1d", "real", s)]["dropped"]]) for s in ASSETS
                     if results[("1d", "real", s)]["dropped"]]
    L = ["# brayvb_v2 DEFAULT preset: cross-asset robustness (Amendment 5)", "",
         "Generated by `research/xasset_brayvb.py`. Zero parameter changes; perp klines, BINANCE_PERP_TAKER + funding, long only; "
         "holdout never loaded (cutoff 2026-03-10 14:00 UTC). BTCUSDT is a reference and is not in any pass rule.", "",
         "## Verdicts", ""] + verdicts + [""]
    L += ["Pass rule (pre-registered): median-over-alts OOS Sharpe > 0 AND Sharpe > 0 on >= 5/7 alts AND Calmar > B&H on >= 4/7 alts. "
          f"Trials: 32 inserted in `research/trials.duckdb` (stage `xasset`; {n_x} xasset rows now); DSR N = {n_before} + 32 + 1 = {N}.", "",
          "## Results per config x asset", "", "\n\n".join(tables), ""]
    L += ["Notes: OOS = stitched daily returns of the retained folds (same calendar windows for all assets, 6-month tiles ending at the cutoff). "
          "B&H is spot-style close-to-close without funding on the same days. Calmar = CAGR / max DD. Dropped folds (train < 12 months): "
          + (", ".join(f"{s}: folds {k}" for s, k in dropped_notes) or "none") + ". Win % counts OOS round trips with pnl > 0.", ""]
    L += ["## Interpretation", "", "@@INTERP@@", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(verdicts))


if __name__ == "__main__":
    main()
