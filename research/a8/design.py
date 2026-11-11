"""Amendment 8 design phase: Optuna per (template, filter, trigger, tf, source), one parameter set for the 8 design coins.

Run:  python research/a8/design.py [--tfs 4h,1d] [--trials 80] [--jobs 3] [--limit N] [--db PATH] [--tag smoke]
Records every trial in trials.duckdb (stage 'a8'). Writes research/reports/a8_design.md and research/runs/a8_candidates.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.a8.core import ROOT, DESIGN, aggregate, eval_coin, gates  # noqa: E402
from research.a8 import templates as T  # noqa: E402
import numpy as np  # noqa: E402

RUNS = ROOT / "research" / "runs"
REPORTS = ROOT / "research" / "reports"
PRUNE_AT, PRUNE_BELOW = 20, 0.0


def fam(c):
    return f"a8:{c['template']}:{c['filter']}:{c['trigger']}:{c['tf']}:{c['source']}"


def trial_hash(spec):
    from btclab.research.search.trials import spec_hash
    return hashlib.sha1((spec_hash(spec) + "|a8|8coins").encode()).hexdigest()


def record(c, spec, a, stage="a8", tag=None):
    h = trial_hash(spec)
    return dict(stage=stage, tf=c["tf"], spec_json=json.dumps(dict(spec, symbols=DESIGN)), spec_hash=h, parent_hash=None,
                family=fam(c), tag=tag or c["template"], fold_json=json.dumps(a["per_coin"]),
                median_oos_sharpe=a["score"], mean_is_sharpe=a["obj"], n_pos_folds=a["n_pos"], is_trades=a["is_trades"],
                min_oos_trades=a["min_trades"], oos_sharpe=a["pool_sharpe"], max_dd=a["pool_dd"], profit_factor=min(a["pf"], 99.0),
                oos_returns=a["pool"].to_numpy(dtype=np.float32).tobytes())


def run_combo(c, n_trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    tf = c["tf"]
    seen, recs = {}, []

    def objective(trial):
        p = T.sample(trial, c)
        spec = T.build(c, p, name="a8")
        h = trial_hash(spec)
        if h in seen:
            a = seen[h][0]
        else:
            try:
                a = aggregate([eval_coin(spec, s, tf) for s in DESIGN])
            except Exception:
                raise optuna.TrialPruned()
            seen[h] = (a, p, spec)
            recs.append(record(c, spec, a))
        trial.set_user_attr("h", h)
        return a["obj"] if np.isfinite(a["obj"]) else -10.0

    def stopper(study, trial):
        done = [t for t in study.trials if t.value is not None]
        if len(study.trials) >= PRUNE_AT and (not done or max(t.value for t in done) < PRUNE_BELOW):
            study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    pruned_early = False
    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[stopper])
    except Exception as e:  # keep what we have
        print("combo error", fam(c), repr(e), flush=True)
    pruned_early = len(study.trials) < n_trials
    out = dict(combo=c, n_trials=len(study.trials), n_unique=len(recs), pruned_early=pruned_early, seconds=time.time() - t0)
    done = [t for t in study.trials if t.value is not None]
    if done:
        best = max(done, key=lambda t: t.value)
        a, p, spec = seen[best.user_attrs["h"]]
        g = gates(a)
        out.update(best=dict(obj=a["obj"], score=a["score"], n_pos=a["n_pos"], pf=a["pf"], side_ok=a["side_ok"],
                             min_trades=a["min_trades"], short_pnl=a["short_pnl"], long_pnl=a["long_pnl"],
                             pool_sharpe=a["pool_sharpe"], pool_dd=a["pool_dd"], per_coin=a["per_coin"]),
                   gates=g, params=p, spec=spec)
    return out, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="4h,1d")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from btclab.research.search.trials import TrialStore
    tfs = args.tfs.split(",")
    todo = T.combos(tfs)
    results_f = RUNS / f"a8_design_combos{args.tag}.jsonl"
    done = set()
    if results_f.exists():
        done = {fam(json.loads(l)["combo"]) for l in results_f.read_text().splitlines() if l.strip()}
    todo = [c for c in todo if fam(c) not in done]
    # 4h and 1d first (order given by --tfs), tf-major
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} combos to run ({len(done)} done)", flush=True)
    store = TrialStore(args.db)
    t0 = time.time()
    from joblib import Parallel, delayed
    gen = Parallel(n_jobs=args.jobs, backend="loky", return_as="generator")(
        delayed(run_combo)(c, args.trials, 1000 + i) for i, c in enumerate(todo))
    n = 0
    for out, recs in gen:
        store.insert_many(recs)
        with open(results_f, "a") as f:
            f.write(json.dumps(out, default=float) + "\n")
        n += 1
        b = out.get("best", {})
        print(f"[{n}/{len(todo)} {time.time() - t0:.0f}s] {fam(out['combo'])} trials={out['n_trials']} "
              f"obj={b.get('obj', float('nan')):.2f} score={b.get('score', float('nan')):.2f} gates={out.get('gates', {}).get('all')}",
              flush=True)
    store.close()
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
