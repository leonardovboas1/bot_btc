"""Amendment 7 validation: V1 (fresh universe), V2 (design coins after the cutoff), V3 (BTC+ETH spot 2017-19).
Run once. Writes research/reports/a7_validation.md and adds stage 'a7' rows to the trials table."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from btclab.research.search.trials import TrialStore  # noqa: E402
from btclab.research.stats.dsr import deflated_sharpe_ratio  # noqa: E402
from research.a7.runner import fmt, run_set, table, verdict  # noqa: E402
from research.a7.strategies import (DESIGN, V2_START, V3_END, V3_START, bench_brayvb,  # noqa: E402
                                    configs, load_panel)

DATA_REPORT = ROOT / "research" / "runs" / "a7_universe_data.md"
UNIVERSE = ROOT / "research" / "runs" / "a7_universe.json"
SETS = ("V1", "V2", "V3")


def main():
    if not DATA_REPORT.exists():
        sys.exit("a7_universe_data.md missing; V1 data not ready")
    v1_syms = [u["symbol"] for u in json.loads(UNIVERSE.read_text())["universe"]]
    results, notes = {}, []

    # V1 ------------------------------------------------------------------
    p1 = load_panel(v1_syms, "perp", allow_holdout=True, need_spot=True)
    btc = load_panel(["BTCUSDT"], "perp", allow_holdout=True)
    br1 = bench_brayvb(list(p1.close.columns), "perp", allow_holdout=True)
    results["V1"] = run_set(p1, br1, btc)
    notes.append(f"V1: {len(p1.close.columns)}/{len(v1_syms)} coins loaded; panel {p1.close.index[0].date()} -> "
                 f"{p1.close.index[-1].date()}; evaluation skips the first 120 days. C3 excludes coins without a spot pair: "
                 f"{', '.join(p1.no_spot) or 'none'}. Coins: {', '.join(p1.close.columns)}.")

    # V2 ------------------------------------------------------------------
    p2 = load_panel(DESIGN, "perp", allow_holdout=True, need_spot=True)
    br2 = bench_brayvb(DESIGN, "perp", allow_holdout=True)
    results["V2"] = run_set(p2, br2, p2, start=V2_START)
    notes.append(f"V2: design coins, evaluated {V2_START.date()} -> {p2.close.index[-1].date()}; signals/weights run on the full "
                 "series (design history = warm-up only), returns sliced at the start.")

    # V3 ------------------------------------------------------------------
    p3 = load_panel(["BTCUSDT", "ETHUSDT"], "spot", V3_START, V3_END, allow_holdout=True)
    br3 = bench_brayvb(["BTCUSDT", "ETHUSDT"], "spot", V3_START, V3_END, allow_holdout=True)
    results["V3"] = run_set(p3, br3, p3, start=V3_START, end=V3_END, only_c1_c2=True)
    notes.append("V3: spot daily, full window 2017-08-17 -> 2019-09-08 (strategies are flat during the 120-day signal warm-up; "
                 "no skip). Spot fee 10 bp + 2 bp slippage, no funding; long-short configs are simulated with synthetic shorts at "
                 "spot cost (historical signal test). brayvb benchmark on spot 4h, SPOT costs, long-only.")

    # verdicts ------------------------------------------------------------
    names = list(configs())
    ver = {}
    for n in names:
        per = {}
        for s in SETS:
            if n in results[s]:
                per[s] = verdict(results[s], n)
        ver[n] = dict(per=per, ok=all(v["ok"] for v in per.values()))

    # DSR ------------------------------------------------------------------
    store = TrialStore()
    n_prior = int(store.con.execute("SELECT count(*) FROM trials WHERE stage != 'a7'").fetchone()[0])
    ledger_var_ann = float(store.con.execute(
        "SELECT var_samp(median_oos_sharpe) FROM trials WHERE stage != 'a7' AND median_oos_sharpe IS NOT NULL").fetchone()[0])
    N = n_prior + 12
    dsr = {}
    for s in SETS:
        sh_pp = np.array([results[s][n]["sharpe"] / np.sqrt(365) for n in names if n in results[s]])
        var_cfg = float(np.nanvar(sh_pp, ddof=1))
        for n in names:
            if n not in results[s]:
                continue
            r = results[s][n]["r"].to_numpy()
            a = deflated_sharpe_ratio(r, n_trials=N, var_trial_sharpes=var_cfg, periods_per_year=365)
            b = deflated_sharpe_ratio(r, n_trials=N, var_trial_sharpes=ledger_var_ann / 365, periods_per_year=365)
            dsr[(s, n)] = (a.dsr, a.expected_max_sr * np.sqrt(365), b.dsr, b.expected_max_sr * np.sqrt(365))

    # trials table ---------------------------------------------------------
    design_file = ROOT / "research" / "runs" / "a7_design_metrics.json"
    design = json.loads(design_file.read_text()) if design_file.exists() else {}
    rows = []
    for s in ("design",) + SETS:
        src = design if s == "design" else results[s]
        for n in names:
            if n not in src:
                continue
            m = src[n]
            spec = dict(stage="a7", config=n, set=s)
            rows.append(dict(stage="a7", tf="1d-a7", spec_json=json.dumps(spec),
                             spec_hash=hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                             family=f"a7:{n.split('_')[0]}", tag=f"{n}:{s}", oos_sharpe=float(m["sharpe"]),
                             max_dd=float(m["mdd"])))
    store.insert_many(rows)
    n_a7 = store.count("a7")
    store.close()

    # report ---------------------------------------------------------------
    best = max((n for n in names), key=lambda n: results["V1"][n]["sharpe"])
    L = ["# A7 validation (Amendment 7)", "", "## Verdict per config (must pass on V1 and V2, and V3 for C1/C2)", "",
         "| config | V1 | V2 | V3 | overall |", "|---|---|---|---|---|"]
    for n in names:
        c = [("PASS" if ver[n]["per"][s]["ok"] else "FAIL") if s in ver[n]["per"] else "n/a" for s in SETS]
        L.append(f"| {n} | {c[0]} | {c[1]} | {c[2]} | **{'PASS' if ver[n]['ok'] else 'FAIL'}** |")
    npass = sum(v["ok"] for v in ver.values())
    L += ["", f"Configs passing: {npass}/12. " + ("None passes: that is the result." if npass == 0 else
          "Passing configs go to paper trading before any live use."), "", "Rule detail:", ""]
    for n in names:
        for s, v in ver[n]["per"].items():
            L.append(f"- {n} {s}: {'PASS' if v['ok'] else 'FAIL'} ({v['detail']})")
    L += ["", f"## Best config (selection across classes, by V1 Sharpe): {best}", ""]
    for s in SETS:
        if best in results[s]:
            m, ew, br = results[s][best], results[s]["EW_BH"], results[s].get("BRAYVB")
            L.append(f"- {s}: Sharpe {fmt(m['sharpe'])}, max DD {fmt(m['mdd'], True)}; EW B&H Sharpe {fmt(ew['sharpe'])} / max DD "
                     f"{fmt(ew['mdd'], True)}; brayvb Sharpe {fmt(br['sharpe']) if br else 'n/a'}.")
    L += ["", "## Deflated Sharpe", "",
          f"N = {n_prior} prior trials + 12 A7 configs = **{N}**. Trial-Sharpe variance (a): across the 12 configs on the same set "
          f"(per-day Sharpes); (b) ledger variance of the prior trials' Sharpes ({ledger_var_ann:.3f}, assumed annualised, /365 per day). "
          "E[max SR] annualised. DSR = probability the true Sharpe exceeds E[max SR]; > 0.95 passes. "
          "Selection of the best config is judged at this N.", "",
          "| set | config | Sharpe | DSR (a) | E[maxSR] (a) | DSR (b) | E[maxSR] (b) |", "|---|---|---|---|---|---|---|"]
    for s in SETS:
        for n in names:
            if (s, n) in dsr:
                a, ea, b, eb = dsr[(s, n)]
                tag = " (best)" if n == best else ""
                L.append(f"| {s} | {n}{tag} | {fmt(results[s][n]['sharpe'])} | {fmt(a, d=3)} | {fmt(ea)} | {fmt(b, d=3)} | {fmt(eb)} |")
    L += ["", f"Trials table: {n_a7} stage-'a7' rows now stored (12 configs x design/V1/V2 + 10 x V3); total ledger "
          f"{n_prior + n_a7} rows.", "", "## Notes", ""] + [f"- {x}" for x in notes]
    for s, title in (("V1", "V1: fresh universe"), ("V2", "V2: design coins after the cutoff"),
                     ("V3", "V3: BTC+ETH spot 2017-08-17 -> 2019-09-08")):
        L += ["", f"## {title}", "", table(results[s])]
    out = ROOT / "research" / "reports" / "a7_validation.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
