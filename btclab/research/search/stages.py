"""Search stages S1-S4 (plan.md section C; docs/preregistration.md section 2.4 + Amendment 1).

S1  single primitive x {long, short} x 3 exits (opposite signal, ATR stop/target, time stop)
S2  survivor entry AND filter primitive (other indicator)
S3  survivor pair AND third primitive, <=300 triples (enumerated to research/runs before running),
    each tuned with Optuna TPE on the IN-SAMPLE (train-window) Sharpe; OOS is only reported
S4  run_gauntlet on the top 50 survivors; DSR with N = every row of the trials table, PBO over the
    family matrix of OOS daily returns, SPA vs B&H, bootstrap, stability, regimes
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from btclab.indicators.registry import REGISTRY
from btclab.research.search import core
from btclab.research.search.core import (SL_ATR, TP_ATR, TIME_STOP, Context, eligible, evaluate_many,
                                         evaluate_spec, get_ctx, top_fraction)
from btclab.research.search.trials import ROOT, TrialStore, spec_hash
from btclab.research.stats import cscv_pbo, deflated_sharpe_ratio, run_gauntlet, trial_sharpes
from btclab.research.stats.gauntlet import GauntletReport, TestResult
from btclab.research.stats.regimes import label_regimes

SEED = 20261110
S3_CAP = 300
S3_PARENTS = 30
FAMILY_CAP = 4000
REPORT_DIR = ROOT / "research" / "reports"
RUNS_DIR = ROOT / "research" / "runs"

# ------------------------------------------------------------------ primitives and S1 enumeration
_SWAP = {}
for _a, _b in [("up", "down"), ("above", "below"), ("bull", "bear"), ("buy", "sell"), ("rising", "falling"),
               ("overbought", "oversold"), ("hband", "lband"), ("upper", "lower"), ("hot", "cold"),
               ("overheated", "undervalued"), ("top", "bottom"), ("green", "red"), ("long", "short"),
               ("high", "low"), ("ge", "le"), ("ob", "os"), ("bullish", "bearish")]:
    _SWAP[_a], _SWAP[_b] = _b, _a


def all_primitives() -> List[str]:
    return [p for s in REGISTRY.values() for p in s.primitives]


def opposite(prim: str) -> Optional[str]:
    """Mirror primitive of the same indicator (cross_above_hband <-> cross_below_lband), or None."""
    ind, _, key = prim.partition(".")
    cand = "_".join(_SWAP.get(t, t) for t in key.split("_"))
    full = f"{ind}.{cand}"
    return full if cand != key and full in REGISTRY[ind].primitives else None


def _base_spec(name: str, tf: str, prim: str) -> dict:
    ind = prim.partition(".")[0]
    return {"name": name, "tf": tf, "indicators": {ind: {}}, "costs": "BINANCE_PERP_TAKER"}


def enumerate_s1(tf: str, prims: Optional[Sequence[str]] = None) -> List[dict]:
    specs = []
    for p in (prims if prims is not None else all_primitives()):
        opp = opposite(p) or f"not:{p}"
        for side in ("long", "short"):
            for style in ("opposite", "atr", "time"):
                s = _base_spec(f"s1|{p}|{side}|{style}", tf, p)
                s[side] = [p]
                if style == "opposite":
                    s[f"exit_{side}"] = [opp]
                elif style == "atr":
                    s["risk"] = {"sl_atr": SL_ATR, "tp_atr": TP_ATR}
                else:
                    s["risk"] = {"time_stop": TIME_STOP.get(tf, 5)}
                specs.append(s)
    return specs


def entry_prims(spec: dict) -> List[str]:
    for side in ("long", "short"):
        if spec.get(side):
            return [p for p in spec[side] if isinstance(p, str)]
    return []


def add_component(parent: dict, prim: str, label: str) -> dict:
    """AND a further primitive onto every entry side of parent (new indicator with default params)."""
    s = copy.deepcopy(parent)
    ind = prim.partition(".")[0]
    s["indicators"].setdefault(ind, {})
    for side in ("long", "short"):
        if s.get(side):
            s[side] = list(s[side]) + [prim]
    s["name"] = f"{label}|{parent['name']}&{prim}"
    return s


def _indicators_of(spec: dict) -> set:
    return set(spec["indicators"])


# ------------------------------------------------------------------ shared runner
class StageResult(dict):
    pass


def _run_specs(store: TrialStore, ctx: Context, tf: str, specs: List[dict], meta: List[dict],
               limit: Optional[int], n_jobs: int, log=print, ctx_kw: Optional[dict] = None) -> dict:
    t0 = time.time()
    done = store.existing_hashes()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(specs))
    if limit:
        order = order[:limit]
    sel = [(specs[i], meta[i]) for i in order]
    seen, todo = set(), []
    for s, m in sel:
        h = spec_hash(s)
        if h in done or h in seen:
            continue
        seen.add(h)
        todo.append((s, m))
    n_new = n_err = 0
    if todo:
        for recs in evaluate_many(ctx, tf, [s for s, _ in todo], [m for _, m in todo], n_jobs, ctx_kw=ctx_kw, log=log):
            good = [r for r in recs if "error" not in r]
            n_err += len(recs) - len(good)
            store.insert_many(good)
            n_new += len(good)
    return dict(n_enumerated=len(specs), n_selected=len(sel), n_new=n_new, n_errors=n_err,
                seconds=time.time() - t0)


def _md_table(df: pd.DataFrame, cols: Sequence[str], fmt: Optional[dict] = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt else (f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _name(spec_json: str) -> str:
    return json.loads(spec_json)["name"]


def write_report(stage: str, tf: str, text: str, report_dir: Optional[Path] = None) -> Path:
    d = Path(report_dir) if report_dir else REPORT_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stage}_{tf}.md"
    p.write_text(text, encoding="utf-8")
    return p


def _stage_report(stage: str, tf: str, ctx: Context, store: TrialStore, info: dict, rows: pd.DataFrame,
                  surv: List[str], extra: str = "") -> str:
    c = ctx.constraints
    n_elig = int(rows.apply(lambda r: eligible(r, c), axis=1).sum()) if len(rows) else 0
    sv = rows[rows.spec_hash.isin(surv)].sort_values("median_oos_sharpe", ascending=False)
    sv = sv.assign(name=sv.spec_json.map(_name))
    fm = {"name": lambda v: f"`{v}`"}
    out = [f"# {stage.upper()} {tf}",
           "",
           f"- Run: {info['n_new']} new trials evaluated ({info['n_errors']} errors) in {info['seconds']:.1f}s "
           f"({info['n_selected']} selected of {info['n_enumerated']} enumerated)",
           f"- Rows in stage/tf: {len(rows)}; eligible (IS trades >= {c['min_is_trades']}, every OOS fold >= "
           f"{c['min_oos_trades']} trades, median OOS Sharpe > 0): {n_elig}; survivors (top 15% of eligible): {len(surv)}",
           f"- Total rows in trials table (DSR N): {store.count()}",
           f"- Folds: {len(ctx.folds)}; OOS days: {int(ctx.oos_mask.sum())}; "
           + (f"first test {ctx.folds[0].test_start.date()}, last test end {ctx.folds[-1].test_end.date()}" if ctx.folds else ""),
           f"- Trade-count minimums for 1d are scaled down from the plan's 100/30 (daily bars cannot reach 30 trades per 6-month fold)."
           if tf == "1d" else "",
           ""]
    if extra:
        out += [extra, ""]
    out += ["## Survivors (top 25 by median OOS Sharpe)", ""]
    if sv.empty:
        out.append("_none_")
    else:
        out.append(_md_table(sv.head(25), ["name", "median_oos_sharpe", "n_pos_folds", "oos_sharpe", "max_dd",
                                           "profit_factor", "is_trades", "min_oos_trades"], fm))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ S1
def run_s1(store: TrialStore, ctx: Context, tf: str, limit=None, n_jobs=1, prims=None, log=print,
           report=True, report_dir=None, ctx_kw=None) -> dict:
    specs = enumerate_s1(tf, prims)
    meta = [dict(stage="s1", tag=entry_prims(s)[0], family="s1") for s in specs]
    info = _run_specs(store, ctx, tf, specs, meta, limit, n_jobs, log, ctx_kw)
    rows = store.fetch("s1", tf)
    surv = top_fraction(rows, ctx.constraints)
    store.set_survivors("s1", tf, surv)
    info.update(survivors=surv, rows=len(rows))
    if report:
        info["report"] = write_report("s1", tf, _stage_report("s1", tf, ctx, store, info, rows, surv), report_dir)
    return info


# ------------------------------------------------------------------ S2
def run_s2(store: TrialStore, ctx: Context, tf: str, limit=None, n_jobs=1, prims=None, log=print,
           report=True, report_dir=None, ctx_kw=None) -> dict:
    parents = store.fetch("s1", tf, survivors=True)
    pool = list(prims) if prims is not None else all_primitives()
    specs, meta = [], []
    for _, pr in parents.iterrows():
        ps = json.loads(pr.spec_json)
        for f in pool:
            if f.partition(".")[0] in _indicators_of(ps):
                continue
            specs.append(add_component(ps, f, "s2"))
            meta.append(dict(stage="s2", parent_hash=pr.spec_hash, tag=f, family="s2"))
    info = _run_specs(store, ctx, tf, specs, meta, limit, n_jobs, log, ctx_kw)
    rows = store.fetch("s2", tf)
    surv = top_fraction(rows, ctx.constraints)
    store.set_survivors("s2", tf, surv)
    info.update(survivors=surv, rows=len(rows))
    if report:
        info["report"] = write_report("s2", tf, _stage_report("s2", tf, ctx, store, info, rows, surv), report_dir)
    return info


# ------------------------------------------------------------------ S3
def _oos(blob) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).astype(np.float64)


def enumerate_triples(store: TrialStore, tf: str, prims=None, cap: int = S3_CAP,
                      n_parents: int = S3_PARENTS) -> List[dict]:
    """Pair survivors x third primitive, ranked by lowest |corr| of the third's best S1 OOS series with
    the pair's OOS series (preregistration 2.4), capped. Written to research/runs before any run."""
    pairs = store.fetch("s2", tf, survivors=True, blobs=True).sort_values("median_oos_sharpe", ascending=False)
    pairs = pairs.head(n_parents)
    s1 = store.fetch("s1", tf, blobs=True)
    best_s1: Dict[str, np.ndarray] = {}
    if len(s1):
        for tag, g in s1.sort_values("median_oos_sharpe", ascending=False).groupby("tag", sort=False):
            best_s1[tag] = _oos(g.iloc[0].oos_returns)
    pool = list(prims) if prims is not None else all_primitives()
    cands = []
    for _, pr in pairs.iterrows():
        ps = json.loads(pr.spec_json)
        po = _oos(pr.oos_returns)
        for t in pool:
            if t.partition(".")[0] in _indicators_of(ps):
                continue
            o = best_s1.get(t)
            if o is not None and len(o) == len(po) and o.std() > 0 and po.std() > 0:
                corr = abs(float(np.corrcoef(po, o)[0, 1]))
            else:
                corr = 1.0
            cands.append((corr, -float(pr.median_oos_sharpe), pr.spec_hash, t))
    cands.sort()
    out = []
    for corr, _, ph, t in cands[:cap]:
        ps = json.loads(pairs[pairs.spec_hash == ph].iloc[0].spec_json)
        out.append(dict(parent_hash=ph, third=t, abs_corr=corr, spec=add_component(ps, t, "s3")))
    return out


def _s3_study(tf: str, kw: dict, spec: dict, n_trials: int, seed: int, family: str, parent_hash: str) -> List[dict]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ctx = get_ctx(tf, kw)
    cache: dict = {}
    seen: Dict[str, dict] = {}

    def sample(trial):
        inds = {}
        for ind in spec["indicators"]:
            p = {}
            for k, sp in REGISTRY[ind].space.items():
                nm = f"{ind}.{k}"
                if sp["type"] == "int":
                    p[k] = trial.suggest_int(nm, sp["low"], sp["high"], step=sp.get("step", 1))
                elif sp["type"] == "float":
                    p[k] = round(trial.suggest_float(nm, sp["low"], sp["high"], step=sp["step"]), 6)
                else:
                    p[k] = trial.suggest_categorical(nm, sp["choices"])
            if not REGISTRY[ind].constraint({**REGISTRY[ind].defaults, **p}):
                raise optuna.TrialPruned()
            inds[ind] = p
        return inds

    def objective(trial):
        s = copy.deepcopy(spec)
        s["indicators"] = sample(trial)
        h = spec_hash(s)
        if h not in seen:
            try:
                if len(cache) > 48:
                    cache.clear()
                rec = evaluate_spec(ctx, s, cache)
            except Exception:
                raise optuna.TrialPruned()
            seen[h] = dict(rec, spec_json=json.dumps(s), spec_hash=h, tf=tf, stage="s3",
                           parent_hash=parent_hash, family=family, tag="s3")
        v = seen[h]["mean_is_sharpe"]
        return v if np.isfinite(v) else -10.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({f"{ind}.{k}": REGISTRY[ind].defaults[k] for ind in spec["indicators"]
                         for k in REGISTRY[ind].space if k in REGISTRY[ind].defaults})
    try:
        study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    except Exception:
        pass
    return list(seen.values())


def run_s3(store: TrialStore, ctx: Context, tf: str, limit=None, n_jobs=1, prims=None, n_trials=200,
           cap=S3_CAP, log=print, report=True, report_dir=None, ctx_kw=None, runs_dir=None) -> dict:
    t0 = time.time()
    triples = enumerate_triples(store, tf, prims, cap)
    rd = Path(runs_dir) if runs_dir else RUNS_DIR
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"combinations_{tf}.json").write_text(json.dumps(
        [dict(parent_hash=t["parent_hash"], third=t["third"], abs_corr=t["abs_corr"], spec=t["spec"]) for t in triples],
        indent=1), encoding="utf-8")
    if limit:
        triples = triples[:limit]
    ctx_kw = ctx_kw or {}
    jobs = []
    for t in triples:
        fam = "s3:" + spec_hash(t["spec"])[:10]
        if store.con.execute("SELECT count(*) FROM trials WHERE family=?", [fam]).fetchone()[0] == 0:
            jobs.append((t, fam))
    n_new = 0
    if jobs:
        if n_jobs == 1 or not ctx.loader:
            core._CTX[tf] = ctx
            results = (_s3_study(tf, ctx_kw, t["spec"], n_trials, SEED + i, fam, t["parent_hash"])
                       for i, (t, fam) in enumerate(jobs))
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                delayed(_s3_study)(tf, ctx_kw, t["spec"], n_trials, SEED + i, fam, t["parent_hash"])
                for i, (t, fam) in enumerate(jobs))
        for i, recs in enumerate(results, 1):
            store.insert_many(recs)
            n_new += len(recs)
            log(f"    triple {i}/{len(jobs)}: {len(recs)} trials")
    # candidate per triple = best IN-SAMPLE config; survivors = top 15% of eligible candidates by OOS Sharpe
    rows = store.fetch("s3", tf)
    cands = rows.sort_values("mean_is_sharpe", ascending=False).groupby("family", sort=False).head(1)
    surv = top_fraction(cands, ctx.constraints)
    store.set_survivors("s3", tf, surv)
    info = dict(n_enumerated=len(triples), n_selected=len(jobs), n_new=n_new, n_errors=0,
                seconds=time.time() - t0, survivors=surv, rows=len(rows))
    if report:
        extra = (f"- Triples enumerated: {len(triples)} (cap {cap}), Optuna TPE trials per triple: {n_trials}; objective = mean "
                 f"in-sample fold Sharpe (OOS never used for tuning). Candidate per triple = best in-sample trial "
                 f"({len(cands)} candidates).\n- Combination list: `research/runs/combinations_{tf}.json`")
        info["report"] = write_report("s3", tf, _stage_report("s3", tf, ctx, store, info, cands, surv, extra), report_dir)
    return info


# ------------------------------------------------------------------ S4
def family_stats(matrix: np.ndarray, best: int, n_total: int, ppy: float = 365.0) -> dict:
    """DSR of column ``best`` with N = n_total trials (trial-Sharpe variance from the family matrix) and PBO (CSCV)."""
    ts = trial_sharpes(matrix)
    dsr = deflated_sharpe_ratio(matrix[:, best], ts, n_trials=n_total, periods_per_year=ppy)
    pbo = cscv_pbo(matrix, n_blocks=16) if matrix.shape[1] >= 2 else None
    return dict(dsr=dsr, pbo=pbo)


def _param_grid(store: TrialStore, family: str):
    g = store.fetch(family=family)
    recs = []
    for _, r in g.iterrows():
        row = {}
        for ind, p in json.loads(r.spec_json)["indicators"].items():
            for k, v in p.items():
                if isinstance(v, (bool, int, float)):
                    row[f"{ind}.{k}"] = float(v)
        row["score"] = r.mean_is_sharpe
        recs.append(row)
    grid = pd.DataFrame(recs)
    cols = [c for c in grid.columns if c != "score"]
    return g, grid, cols


def run_s4(store: TrialStore, ctx: Context, tf: str, top_n: int = 50, log=print, report=True, report_dir=None,
           n_boot_paths: int = 10_000, spa_boot: int = 2000) -> dict:
    t0 = time.time()
    from btclab.signals.combiner import run_spec
    n_total = store.count()
    pool = store.fetch(tf=tf, survivors=True)
    pool = pool[pool.apply(lambda r: eligible(r, ctx.constraints), axis=1)]
    pool = pool.sort_values("median_oos_sharpe", ascending=False).head(top_n)
    # family matrix: all trials of this tf with an OOS series (random subsample if too big) + the candidates
    all_h = store.con.execute("SELECT spec_hash FROM trials WHERE tf=? AND oos_returns IS NOT NULL", [tf]).fetchall()
    all_h = [h[0] for h in all_h]
    cand_h = pool.spec_hash.tolist()
    if len(all_h) > FAMILY_CAP:
        rng = np.random.default_rng(SEED)
        keep = set(rng.choice(all_h, FAMILY_CAP, replace=False).tolist()) | set(cand_h)
        all_h = [h for h in all_h if h in keep]
    matrix, hs = store.returns_matrix(all_h)
    if matrix.size:
        sd = matrix.std(axis=0)
        ok = sd > 0
        matrix, hs = matrix[:, ok], [h for h, k in zip(hs, ok) if k]
    pbo = cscv_pbo(matrix, n_blocks=16) if matrix.shape[1] >= 2 else None
    ts = trial_sharpes(matrix) if matrix.shape[1] >= 2 else np.array([0.0, 1e-3])
    oos_days = ctx.oos_days
    bench = pd.Series(ctx.bench[ctx.oos_mask], index=oos_days)
    reg = label_regimes(ctx.close_daily).reindex(oos_days)
    out_rows = []
    for _, r in pool.iterrows():
        spec = json.loads(r.spec_json)
        res, _ = run_spec(spec, ctx.df)
        rec = core.summarize(ctx, res)
        dr = (core.daily_returns(res.equity, ctx.df.index).reindex(ctx.days).fillna(0.0)).to_numpy()
        oos = pd.Series(dr[ctx.oos_mask], index=oos_days)
        folds = [dr[m] for m in ctx.test_masks]
        kw = {}
        if r.stage == "s3":
            g, grid, cols = _param_grid(store, r.family)
            if len(grid) > 3 and cols:
                kw = dict(param_grid=grid, param_cols=cols, score_col="score",
                          peak_index=int(g.index[g.spec_hash == r.spec_hash][0]))
        rep = run_gauntlet(
            oos, n_trials=n_total, benchmark_returns=bench, oos_fold_returns=folds, regimes=reg,
            n_is_trades=rec["is_trades"], oos_fold_trade_counts=[f["oos_trades"] for f in json.loads(rec["fold_json"])],
            max_drawdown=rec["max_dd"], profit_factor=rec["profit_factor"],
            criteria=dict(spa_n_boot=spa_boot, bootstrap_paths=n_boot_paths), seed=0, **kw)
        rep = _patch_family(rep, oos.to_numpy(), ts, n_total, pbo)
        out_rows.append(dict(spec_hash=r.spec_hash, name=spec["name"], stage=r.stage,
                             median_oos_sharpe=r.median_oos_sharpe, report=rep))
        log(f"    {spec['name'][:60]}: {rep.n_passed} pass / {rep.n_failed} fail")
    info = dict(n_total=n_total, family_size=int(matrix.shape[1]), pbo=None if pbo is None else pbo.pbo,
                rows=out_rows, seconds=time.time() - t0, n_candidates=len(out_rows))
    if report:
        info["report"] = write_report("s4", tf, _s4_report(tf, ctx, store, info), report_dir)
    return info


def _patch_family(rep: GauntletReport, oos: np.ndarray, ts: np.ndarray, n_total: int, pbo) -> GauntletReport:
    """Replace the single-candidate DSR / missing PBO by the family versions (N = all DB trials)."""
    res = dict(rep.results)
    d = deflated_sharpe_ratio(oos, ts, n_trials=n_total, periods_per_year=365.0, threshold=0.95)
    res["dsr"] = TestResult("dsr", d.dsr, "> 0.95", d.passed, d.to_dict())
    if pbo is not None:
        res["pbo"] = TestResult("pbo", pbo.pbo, "< 0.2", pbo.passed, {"n_splits": pbo.n_splits})
    ex = [t for t in res.values() if t.passed is not None]
    npass = sum(t.passed for t in ex)
    return GauntletReport(res, npass == len(ex) and len(ex) > 0, npass, len(ex) - npass, len(res) - len(ex), rep.holdout)


def _prereg_hash() -> str:
    p = ROOT / "docs" / "preregistration.md"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "n/a"


def _s4_report(tf: str, ctx: Context, store: TrialStore, info: dict) -> str:
    tests = ["wf_oos_sharpe", "dsr", "pbo", "spa_vs_zero", "spa_vs_bh", "bootstrap", "stability", "regimes",
             "min_is_trades", "min_oos_trades", "max_drawdown", "profit_factor"]
    out = [f"# S4 gauntlet {tf}", "",
           f"- preregistration.md sha256: `{_prereg_hash()}`",
           f"- DSR N (all rows in trials table, all stages and TFs): {info['n_total']}",
           f"- Family matrix for PBO/DSR variance: {info['family_size']} trials of this TF; PBO (CSCV, S=16): "
           + ("n/a" if info["pbo"] is None else f"{info['pbo']:.3f} (pass < 0.20)"),
           f"- Candidates: {info['n_candidates']} (top 50 eligible survivors by median OOS Sharpe); runtime {info['seconds']:.1f}s",
           "- Holdout untouched (loaders capped at the cutoff).", ""]
    if not info["rows"]:
        return "\n".join(out + ["_no eligible survivors_"]) + "\n"
    out += ["| strategy | stage | med OOS Sharpe | all pass | " + " | ".join(tests) + " |",
            "|" + "---|" * (4 + len(tests))]
    for r in info["rows"]:
        rep = r["report"]
        cells = [rep.results[t].status if t in rep.results else "-" for t in tests]
        out.append(f"| `{r['name'][:70]}` | {r['stage']} | {r['median_oos_sharpe']:.2f} | "
                   f"{'YES' if rep.passed else 'no'} | " + " | ".join(cells) + " |")
    n_all = sum(r["report"].passed for r in info["rows"])
    out += ["", f"Candidates passing every executed test: {n_all} of {len(info['rows'])}."]
    return "\n".join(out) + "\n"


STAGES = ("s1", "s2", "s3", "s4")
