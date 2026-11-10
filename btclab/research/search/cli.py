"""CLI: python -m btclab.research.search --stage s1 --tf 4h [--limit N]"""
from __future__ import annotations

import argparse
import time

from btclab.research.search import stages
from btclab.research.search.core import build_context, n_jobs_default
from btclab.research.search.trials import TrialStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="btclab.research.search")
    ap.add_argument("--stage", required=True, choices=["s1", "s2", "s3", "s4", "all"])
    ap.add_argument("--tf", default="all", help="1h, 4h, 1d, comma list, or all")
    ap.add_argument("--limit", type=int, default=None, help="max configs (S1/S2) or triples (S3) per run")
    ap.add_argument("--n-jobs", type=int, default=None, help="default cpu_count-1")
    ap.add_argument("--db", default=None, help="trials.duckdb path (default research/trials.duckdb)")
    ap.add_argument("--s3-trials", type=int, default=200, help="Optuna trials per triple")
    ap.add_argument("--s3-cap", type=int, default=stages.S3_CAP)
    ap.add_argument("--top", type=int, default=50, help="S4 candidates")
    a = ap.parse_args(argv)

    tfs = ["1h", "4h", "1d"] if a.tf == "all" else a.tf.split(",")
    todo = list(stages.STAGES) if a.stage == "all" else [a.stage]
    n_jobs = a.n_jobs or n_jobs_default()
    store = TrialStore(a.db)
    try:
        for tf in tfs:
            ctx = build_context(tf)
            for st in todo:
                t0 = time.time()
                print(f"[{st} {tf}] start (n_jobs={n_jobs})", flush=True)
                if st == "s1":
                    info = stages.run_s1(store, ctx, tf, a.limit, n_jobs)
                elif st == "s2":
                    info = stages.run_s2(store, ctx, tf, a.limit, n_jobs)
                elif st == "s3":
                    info = stages.run_s3(store, ctx, tf, a.limit, n_jobs, n_trials=a.s3_trials, cap=a.s3_cap)
                else:
                    info = stages.run_s4(store, ctx, tf, a.top)
                print(f"[{st} {tf}] done in {time.time() - t0:.1f}s; report: {info.get('report')}", flush=True)
    finally:
        store.close()
    return 0
