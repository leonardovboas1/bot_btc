"""Honest tuning of brayvb_v2 (Gaussian Channel + Stoch RSI, long only).

Nested walk-forward: for each of the 6 WF folds, Optuna TPE tunes on that fold's TRAINING window only,
then that fold's winner is scored on the untouched test window. Concatenated test results = honest OOS
estimate of "tuning brayvb". Final config per tf x source = tuned on ALL in-sample data (holdout never loaded).

Usage:
  python research/brayvb_tune.py worker 1h:real,4h:ha [--trials 300]   # tuning (CPU heavy, pin cores!)
  python research/brayvb_tune.py finalize                                # trials table, stats, report
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btclab.api.presets import get_preset
from btclab.indicators.registry import REGISTRY
from btclab.engine.metrics import daily_returns
from btclab.research.search.core import PPY, _max_dd, _sh, build_context, summarize
from btclab.research.search.trials import TrialStore, spec_hash
from btclab.signals.combiner import run_spec

RUNS = ROOT / "research" / "runs" / "brayvb_tune"
REPORT = ROOT / "research" / "reports" / "brayvb_tune.md"
DB = ROOT / "research" / "trials_brayvb_tune.duckdb"
TFS, SOURCES = ("1h", "4h", "1d"), ("real", "ha")
SEED = 20261110
SL_VALUES = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]          # 0 = off
TRAIL_VALUES = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
INDS = ("gc", "stochrsi")


# ---------------------------------------------------------------- space
def dims():
    """Ordered dict name -> list of values (ordering gives +-1 step neighbours)."""
    d = {}
    for ind in INDS:
        for k, sp in REGISTRY[ind].space.items():
            if sp["type"] == "cat":
                vals = list(sp["choices"])
            elif sp["type"] == "int":
                vals = list(range(sp["low"], sp["high"] + 1, sp.get("step", 1)))
            else:
                n = int(round((sp["high"] - sp["low"]) / sp["step"]))
                vals = [round(sp["low"] + i * sp["step"], 6) for i in range(n + 1)]
            d[f"{ind}.{k}"] = vals
    d["risk.sl_atr"] = SL_VALUES
    d["risk.trail_atr"] = TRAIL_VALUES
    return d


DIMS = dims()


def default_cfg():
    c = {}
    for ind in INDS:
        for k in REGISTRY[ind].space:
            c[f"{ind}.{k}"] = REGISTRY[ind].defaults[k]
    c["risk.sl_atr"], c["risk.trail_atr"] = 0.0, 0.0
    return c


def ckey(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def build_spec(tf, source, cfg):
    s = get_preset("brayvb_v2")
    s["tf"] = tf
    s["indicators"] = {ind: {k.split(".", 1)[1]: v for k, v in cfg.items() if k.startswith(ind + ".")} for ind in INDS}
    s["risk"] = {k.split(".", 1)[1]: v for k, v in cfg.items() if k.startswith("risk.") and v}
    if source != "real":
        s["source"] = source
    return s


def neighbours(cfg):
    out = []
    for name, vals in DIMS.items():
        if len(vals) < 2:
            continue
        i = vals.index(cfg[name])
        if isinstance(vals[0], bool):
            c = dict(cfg); c[name] = vals[1 - i]; out.append((name, c))
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                c = dict(cfg); c[name] = vals[j]; out.append((name, c))
    return out


# ---------------------------------------------------------------- evaluation
class Evaluator:
    def __init__(self, tf, source):
        self.tf, self.source = tf, source
        self.ctx = build_context(tf)                  # loaders enforce the holdout cutoff
        self.cache, self.seen = {}, {}
        c = self.ctx
        t0 = pd.Timestamp(c.folds[0].train_start)
        t0 = t0.tz_convert("UTC").tz_localize(None) if t0.tzinfo is not None else t0
        self.all_mask = c.days >= t0.normalize()
        n_all = int(self.all_mask.sum())
        self.min_is = c.constraints["min_is_trades"]
        # fold training windows are shorter than the whole IS span: scale the A2 minimum by window length
        self.min_train = [max(5, math.ceil(self.min_is * trm.sum() / n_all)) for trm in c.train_masks]
        self.train_rng = [(int(f.train_idx[0]), int(f.train_idx[-1])) for f in c.folds]

    def eval(self, cfg, tag):
        key = ckey(cfg)
        if key in self.seen:
            return self.seen[key]
        spec = build_spec(self.tf, self.source, cfg)
        if len(self.cache) > 48:
            self.cache.clear()
        res, _ = run_spec(spec, self.ctx.df, self.cache)
        rec = summarize(self.ctx, res)
        folds = json.loads(rec["fold_json"])
        dr = daily_returns(res.equity, self.ctx.df.index).reindex(self.ctx.days).fillna(0.0).to_numpy()
        e = np.asarray(res.trades["entry_idx"])
        trn = [int(((e >= lo) & (e <= hi)).sum()) for lo, hi in self.train_rng]
        r = dict(cfg=cfg, spec=spec, rec=rec, folds=folds, dr=dr.astype(np.float32), entry=e,
                 pnl=np.asarray(res.trades["pnl"]), train_trades=trn, tag=tag,
                 all_sharpe=_sh(dr[self.all_mask]), all_trades=int(len(e)))
        self.seen[key] = r
        return r


def penalty_obj(sh, n, need):
    if n < need:
        return -5.0 + n / need
    return sh if np.isfinite(sh) else -10.0


def tune(ev: Evaluator, fold, n_trials, seed):
    """fold = int (train window of that fold) or None (all in-sample). Returns (best cfg, best value)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tag = "final" if fold is None else f"fold{fold}"

    def objective(trial):
        cfg = {n: trial.suggest_categorical(n, v) for n, v in DIMS.items()}
        r = ev.eval(cfg, tag)
        if fold is None:
            return penalty_obj(r["all_sharpe"], r["all_trades"], ev.min_is)
        return penalty_obj(r["folds"][fold]["is_sharpe"], r["train_trades"][fold], ev.min_train[fold])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({n: min(DIMS[n], key=lambda x: abs(float(x) - float(v))) if not isinstance(v, bool) else v
                         for n, v in default_cfg().items()})
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    return dict(study.best_params), study.best_value


def run_group(tf, source, n_trials, log):
    t0 = time.time()
    ev = Evaluator(tf, source)
    nf = len(ev.ctx.folds)
    out = dict(tf=tf, source=source, n_folds=nf, min_train=ev.min_train, folds=[])
    ev.eval(default_cfg(), "default")
    for k in range(nf):
        best, val = tune(ev, k, n_trials, SEED + k)
        r = ev.eval(best, f"fold{k}")
        out["folds"].append(dict(cfg=best, train_obj=val, train_sharpe=r["folds"][k]["is_sharpe"],
                                 test_sharpe=r["folds"][k]["oos_sharpe"], test_ret=r["folds"][k]["oos_ret"],
                                 test_trades=r["folds"][k]["oos_trades"]))
        log(f"  {tf}:{source} fold{k} train={val:.2f} test={r['folds'][k]['oos_sharpe']:.2f} ({time.time()-t0:.0f}s)")
    best, val = tune(ev, None, n_trials, SEED + 99)
    ev.eval(best, "final")
    out["final_cfg"], out["final_obj"] = best, val
    log(f"  {tf}:{source} final IS={val:.2f} ({time.time()-t0:.0f}s)")
    # +-1 step neighbours of the final config (plateau vs spike); evaluated => they are trials too
    out["neigh"] = []
    for name, c in neighbours(best):
        r = ev.eval(c, "neigh")
        out["neigh"].append(dict(dim=name, cfg=c, sharpe=r["all_sharpe"], trades=r["all_trades"]))
    out["trials"] = list(ev.seen.values())
    out["default_key"] = ckey(default_cfg())
    out["test_masks"] = ev.ctx.test_masks
    out["test_bars"] = ev.ctx.test_bars
    out["oos_mask"] = ev.ctx.oos_mask
    out["fold_windows"] = [(str(f.train_start.date()), str(f.train_end.date()), str(f.test_start.date()),
                            str(f.test_end.date())) for f in ev.ctx.folds]
    out["seconds"] = time.time() - t0
    out["n_evals"] = len(ev.seen)
    return out


def worker(groups, n_trials):
    RUNS.mkdir(parents=True, exist_ok=True)
    for g in groups:
        tf, source = g.split(":")
        f = RUNS / f"{tf}_{source}.pkl"
        if f.exists():
            print("skip", g); continue
        res = run_group(tf, source, n_trials, lambda m: print(m, flush=True))
        with open(f, "wb") as fh:
            pickle.dump(res, fh)
        print(f"done {g}: {res['n_evals']} evals, {res['seconds']:.0f}s", flush=True)


# ---------------------------------------------------------------- finalize
def series_metrics(r, trades_pnl):
    r = np.asarray(r, float)
    wins = trades_pnl > 0
    return dict(sharpe=_sh(r), ret=float(np.prod(1 + r) - 1), mdd=_max_dd(r),
                win=float(wins.mean()) if len(wins) else float("nan"), trades=int(len(trades_pnl)))


def nested_series(res, cfgs):
    """Concatenate test-window daily returns + trade pnl of per-fold configs."""
    by = {ckey(t["cfg"]): t for t in res["trials"]}
    rs, pn = [], []
    for k, c in enumerate(cfgs):
        t = by[ckey(c)]
        rs.append(t["dr"].astype(float)[res["test_masks"][k]])
        lo, hi = res["test_bars"][k]
        m = (t["entry"] >= lo) & (t["entry"] <= hi)
        pn.append(t["pnl"][m])
    return np.concatenate(rs), np.concatenate(pn)


def main_N():
    """Row count of the main trials table (copied, since the live search holds the lock)."""
    import shutil, tempfile, duckdb
    d = Path(tempfile.mkdtemp())
    try:
        for suf in ("", ".wal"):
            s = ROOT / "research" / f"trials.duckdb{suf}"
            if s.exists():
                shutil.copy(s, d / f"t.duckdb{suf}")
        c = duckdb.connect(str(d / "t.duckdb"))
        return int(c.execute("select count(*) from trials").fetchone()[0])
    except Exception as e:
        print("main N unavailable:", e)
        return 0


def fmt_cfg(c):
    parts = []
    for n, v in c.items():
        if n.startswith("risk."):
            if v:
                parts.append(f"{n.split('.')[1]}={v}")
        else:
            parts.append(f"{n.split('.', 1)[1]}={v}")
    return ", ".join(parts)


def finalize():
    from btclab.research.stats import cscv_pbo, deflated_sharpe_ratio, trial_sharpes
    if DB.exists():
        DB.unlink()
    store = TrialStore(DB)
    n_main = main_N()
    res_all = {}
    for tf in TFS:
        for src in SOURCES:
            f = RUNS / f"{tf}_{src}.pkl"
            if f.exists():
                res_all[(tf, src)] = pickle.load(open(f, "rb"))
    for (tf, src), res in res_all.items():
        recs = [dict(t["rec"], spec_json=json.dumps(t["spec"]), spec_hash=spec_hash(t["spec"]), tf=tf,
                     stage="brayvb_tune", parent_hash=None, family=f"brayvb_tune:{tf}:{src}", tag=t["tag"])
                for t in res["trials"]]
        store.insert_many(recs)
    n_own = store.count()
    n_total = n_main + n_own
    summ = []
    for (tf, src), res in res_all.items():
        fam = f"brayvb_tune:{tf}:{src}"
        by = {ckey(t["cfg"]): t for t in res["trials"]}
        dflt = by[res["default_key"]]
        nf = res["n_folds"]
        nest_r, nest_p = nested_series(res, [f["cfg"] for f in res["folds"]])
        dflt_r, dflt_p = nested_series(res, [default_cfg()] * nf)
        nm, dm = series_metrics(nest_r, nest_p), series_metrics(dflt_r, dflt_p)
        fin = by[ckey(res["final_cfg"])]
        fh = [h for (h,) in store.con.execute(
            "SELECT spec_hash FROM trials WHERE family=? AND oos_returns IS NOT NULL ORDER BY id", [fam]).fetchall()]
        mat, hs = store.returns_matrix(fh)
        ok = mat.std(axis=0) > 0
        mat, hs = mat[:, ok], [h for h, k in zip(hs, ok) if k]
        if len(hs) > 2500:
            rng = np.random.default_rng(SEED)
            keep = set(rng.choice(hs, 2500, replace=False).tolist()) | {spec_hash(fin["spec"])}
            sel = [i for i, h in enumerate(hs) if h in keep]
            mat, hs = mat[:, sel], [hs[i] for i in sel]
        ts = trial_sharpes(mat)
        pbo = cscv_pbo(mat, n_blocks=16) if mat.shape[1] >= 2 else None
        fin_oos = fin["dr"].astype(float)[res["oos_mask"]]
        d_nested = deflated_sharpe_ratio(nest_r, ts, n_trials=n_total, periods_per_year=PPY)
        d_final = deflated_sharpe_ratio(fin_oos, ts, n_trials=n_total, periods_per_year=PPY)
        d_dflt = deflated_sharpe_ratio(dflt_r, ts, n_trials=n_total, periods_per_year=PPY)
        nb = np.array([x["sharpe"] for x in res["neigh"]]) if res["neigh"] else np.array([np.nan])
        win_cfgs = pd.DataFrame([f["cfg"] for f in res["folds"]])
        agree = {c: (win_cfgs[c].mode().iloc[0], float((win_cfgs[c] == win_cfgs[c].mode().iloc[0]).mean()))
                 for c in win_cfgs.columns}
        summ.append(dict(tf=tf, src=src, res=res, nm=nm, dm=dm, fin=fin, fm=series_metrics(fin_oos, fin["pnl"]),
                         dflt=dflt, pbo=pbo, d_nested=d_nested, d_final=d_final, d_dflt=d_dflt, nb=nb,
                         agree=agree, n_fam=mat.shape[1]))
    L = ["# brayvb_v2 tuning (Gaussian Channel + Stoch RSI, long only)", "",
         "Generated by `research/brayvb_tune.py`. BTCUSDT perp, BINANCE_PERP_TAKER costs, funding included, holdout never loaded.",
         f"Trials table: `research/trials_brayvb_tune.duckdb` ({n_own} unique configs, families `brayvb_tune:<tf>:<source>`). "
         f"The live search holds the main `research/trials.duckdb` ({n_main} rows at read time), so DSR N = {n_main} + {n_own} = {n_total}.", "",
         "Method: per WF fold, Optuna TPE (seeded, default preset enqueued as trial 0) maximises TRAIN-window Sharpe "
         "(min trades: A2 in-sample minimum scaled by train-window length, soft penalty); the winner is scored on the untouched test window. "
         "Nested OOS = concatenated test windows of the six fold winners. Default = the unmodified preset on the same folds. "
         "Final config = tuned on all in-sample data (its OOS columns are therefore NOT out-of-sample).", "",
         "## 1. Honest estimate of tuning (nested walk-forward OOS)", "",
         "| tf | src | tuned Sharpe | tuned ret | tuned maxDD | tuned win% | tuned trades | default Sharpe | default ret | default maxDD | default trades | folds tuned>default | DSR p tuned | DSR p default |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        res = s["res"]
        fb = sum(f["test_sharpe"] > s["dflt"]["folds"][k]["oos_sharpe"] for k, f in enumerate(res["folds"]))
        L.append(f"| {s['tf']} | {s['src']} | {s['nm']['sharpe']:.2f} | {s['nm']['ret']*100:.0f}% | {s['nm']['mdd']*100:.0f}% | {s['nm']['win']*100:.0f}% | {s['nm']['trades']} | "
                 f"{s['dm']['sharpe']:.2f} | {s['dm']['ret']*100:.0f}% | {s['dm']['mdd']*100:.0f}% | {s['dm']['trades']} | {fb}/{res['n_folds']} | "
                 f"{s['d_nested'].dsr:.3f} | {s['d_dflt'].dsr:.3f} |")
    L += ["", "## 2. Final configs (tuned on all in-sample data)", "",
          "| tf | src | params | IS Sharpe | OOS-window Sharpe* | ret* | maxDD* | win%* | trades (all IS) | neighbour Sharpe mean (min) | PBO | DSR p* |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        fm = s["fm"]
        L.append(f"| {s['tf']} | {s['src']} | {fmt_cfg(s['res']['final_cfg'])} | {s['fin']['all_sharpe']:.2f} | {fm['sharpe']:.2f} | {fm['ret']*100:.0f}% | "
                 f"{fm['mdd']*100:.0f}% | {fm['win']*100:.0f}% | {s['fin']['all_trades']} | {np.nanmean(s['nb']):.2f} ({np.nanmin(s['nb']):.2f}) | "
                 f"{s['pbo'].pbo:.2f} | {s['d_final'].dsr:.3f} |")
    L += ["", "\\* final config evaluated on the six test windows, which were part of its tuning data: optimistic, reference only. "
          "PBO = CSCV, 16 blocks, over all configs of the family. DSR p: N = total trials above, trial-Sharpe variance from the family matrix.", "",
          "## 3. Per-fold winners and parameter stability", ""]
    for s in summ:
        res = s["res"]
        L += [f"### {s['tf']} / {s['src']}  (PBO matrix columns: {s['n_fam']}; default IS Sharpe {s['dflt']['all_sharpe']:.2f})", "",
              "| fold | train / test window | train Sharpe | test Sharpe | default test Sharpe | test trades | winner |", "|---|---|---|---|---|---|---|"]
        for k, f in enumerate(res["folds"]):
            w = res["fold_windows"][k]
            L.append(f"| {k} | {w[0]}..{w[1]} / {w[2]}..{w[3]} | {f['train_sharpe']:.2f} | {f['test_sharpe']:.2f} | {s['dflt']['folds'][k]['oos_sharpe']:.2f} | {f['test_trades']} | {fmt_cfg(f['cfg'])} |")
        ag = ", ".join(f"{k}={v[0]} ({v[1]*100:.0f}%)" for k, v in s["agree"].items() if v[1] >= 0.5)
        L += ["", f"Winner agreement (params where >=50% of fold winners share the modal value; risk 0 = off): {ag or 'none'}.",
              f"Neighbour (+-1 step) Sharpe of the final config: mean {np.nanmean(s['nb']):.2f}, min {np.nanmin(s['nb']):.2f} "
              f"vs {s['fin']['all_sharpe']:.2f} at the optimum ({len(s['nb'])} neighbours).", ""]
    best = max(summ, key=lambda s: s["nm"]["sharpe"] - s["dm"]["sharpe"])
    best_abs = max(summ, key=lambda s: s["nm"]["sharpe"])
    L += ["## 4. Verdict", "",
          f"Best nested-OOS procedure: {best_abs['tf']}/{best_abs['src']} (Sharpe {best_abs['nm']['sharpe']:.2f}, DSR p {best_abs['d_nested'].dsr:.3f}). "
          f"Largest gain over default: {best['tf']}/{best['src']} ({best['nm']['sharpe']:.2f} vs {best['dm']['sharpe']:.2f}).", "",
          "VERDICT_PLACEHOLDER"]
    json.dump({f"{s['tf']}:{s['src']}": dict(final_cfg=s["res"]["final_cfg"], spec=s["fin"]["spec"], nested=s["nm"],
                                          default=s["dm"], dsr_nested=s["d_nested"].dsr, pbo=s["pbo"].pbo,
                                          is_sharpe=s["fin"]["all_sharpe"]) for s in summ},
              open(ROOT / "research" / "reports" / "brayvb_tune_configs.json", "w"), indent=1, default=str)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("report written", REPORT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worker", "finalize"])
    ap.add_argument("groups", nargs="?", default="")
    ap.add_argument("--trials", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "worker":
        worker([g for g in a.groups.split(",") if g], a.trials)
    else:
        finalize()
