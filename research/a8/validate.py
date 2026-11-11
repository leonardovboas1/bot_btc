"""Amendment 8 validation: the advancing candidates are run ONCE on V1 (25 fresh coins) and V2 (8 design coins after the cutoff).
Writes research/reports/a8_validation.md, research/runs/a8_validation.json and appends 'a8val' trials.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.a8.core import ROOT, DESIGN, run_fast  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from btclab.api.presets import get_preset  # noqa: E402
from btclab.data.loaders import get_cutoff, load_funding, load_ohlcv  # noqa: E402
from btclab.engine.metrics import daily_returns  # noqa: E402

PPY = 365.0
WARM_DAYS = 120
FRESH = [x["symbol"] for x in json.loads((ROOT / "research/runs/a7_universe.json").read_text())["universe"]]


def naive_days(ix):
    ix = pd.DatetimeIndex(ix)
    return (ix.tz_convert("UTC").tz_localize(None) if ix.tz is not None else ix).normalize()


def load(sym, tf):
    df = load_ohlcv("perp", tf, allow_holdout=True, symbol=sym)
    f = load_funding(tf, allow_holdout=True, symbol=sym).reindex(df.index).fillna(0.0)
    return df.assign(funding=f.to_numpy())


def sharpe(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1) if len(r) > 2 else 0
    return float(r.mean() / sd * np.sqrt(PPY)) if sd > 0 else float("nan")


def mdd(r):
    eq = np.cumprod(1 + np.asarray(r, float))
    return float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else float("nan")


def pf(pnl):
    w, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (99.0 if w > 0 else 0.0)


def run_on(spec, df, start_bar):
    """-> (daily strat returns Series, B&H daily returns Series, trade pnl array, sides) for bars >= start_bar."""
    res = run_fast(spec, df, {})
    idx = df.index
    eq = np.asarray(res.equity, float)
    base = eq[start_bar - 1] if start_bar > 0 else eq[0]
    dr = daily_returns(eq[start_bar:], idx[start_bar:], init_equity=base)
    dr.index = naive_days(dr.index)
    close = df["close"].to_numpy(float)
    prev = np.r_[close[0], close[:-1]]
    bar = (close / prev - 1 - df["funding"].to_numpy(float))[start_bar:]
    bh = daily_returns(np.cumprod(1 + bar), idx[start_bar:], init_equity=1.0)
    bh.index = naive_days(bh.index)
    tr = res.trades
    ent = np.asarray(tr["entry_idx"])
    sel = ent >= start_bar
    return dr, bh, np.asarray(tr["pnl"])[sel], np.asarray(tr["side"])[sel]


def evaluate_set(spec, syms, tf, mode):
    cutoff = get_cutoff()
    sr, br, pn, sd, per = {}, {}, [], [], {}
    for s in syms:
        df = load(s, tf)
        if mode == "V2":
            cut = cutoff.tz_convert(df.index.tz) if df.index.tz is not None else cutoff.tz_convert("UTC").tz_localize(None)
            sb = int(df.index.searchsorted(cut))
        else:
            sb = int(df.index.searchsorted(df.index[0] + pd.Timedelta(days=WARM_DAYS)))
        if len(df) - sb < 60:
            continue
        dr, bh, p, sdd = run_on(spec, df, sb)
        sr[s], br[s] = dr, bh
        pn.append(p)
        sd.append(sdd)
        per[s] = sharpe(dr.to_numpy())
    S = pd.DataFrame(sr)
    B = pd.DataFrame(br)
    ew, ewb = S.mean(axis=1, skipna=True), B.mean(axis=1, skipna=True)
    pnl = np.concatenate(pn)
    sides = np.concatenate(sd)
    sh = np.array(list(per.values()))
    return dict(n_coins=len(per), med_sharpe=float(np.nanmedian(sh)), frac_pos=float(np.nanmean(sh > 0)), pf=pf(pnl),
                short_pnl=float(pnl[sides < 0].sum()), long_pnl=float(pnl[sides > 0].sum()), trades=int(len(pnl)),
                ew_sharpe=sharpe(ew.to_numpy()), ew_dd=mdd(ew.to_numpy()), ew_cagr=float(np.prod(1 + ew.to_numpy()) ** (PPY / len(ew)) - 1),
                bh_sharpe=sharpe(ewb.to_numpy()), bh_dd=mdd(ewb.to_numpy()), bh_cagr=float(np.prod(1 + ewb.to_numpy()) ** (PPY / len(ewb)) - 1),
                per_coin=per, series=ew)


def verdict(r, v1):
    a = r["med_sharpe"] > 0.5
    b = r["frac_pos"] >= 0.6
    c = r["pf"] > 1.15
    d = r["ew_dd"] <= 0.7 * r["bh_dd"]
    e = (r["ew_sharpe"] >= r["bh_sharpe"]) if v1 else True
    return dict(med=a, pos=b, pf=c, dd=d, vs_bh=e, ok=bool(a and b and c and d and e))


def main():
    t0 = time.time()
    cands = json.loads((ROOT / "research/runs/a8_candidates.json").read_text())
    brayvb = get_preset("brayvb_v2")
    brayvb["name"] = "brayvb_v2"
    out = []
    bray = {}
    for mode, syms in (("V1", FRESH), ("V2", DESIGN)):
        bray[mode] = evaluate_set(brayvb, syms, "4h", mode)
        print(mode, "brayvb", round(bray[mode]["ew_sharpe"], 2), flush=True)
    for c in cands:
        spec, tf = c["spec"], c["combo"]["tf"]
        rec = dict(cand=c)
        for mode, syms in (("V1", FRESH), ("V2", DESIGN)):
            r = evaluate_set(spec, syms, tf, mode)
            r["verdict"] = verdict(r, mode == "V1")
            rec[mode] = r
        rec["pass"] = bool(rec["V1"]["verdict"]["ok"] and rec["V2"]["verdict"]["ok"])
        out.append(rec)
        print(c["name"], "V1", rec["V1"]["verdict"]["ok"], round(rec["V1"]["ew_sharpe"], 2), "V2", rec["V2"]["verdict"]["ok"],
              round(rec["V2"]["ew_sharpe"], 2), flush=True)

    # trial accounting + DSR at updated N
    from btclab.research.search.trials import TrialStore
    from btclab.research.stats import deflated_sharpe_ratio
    st = TrialStore()
    import hashlib
    rows = []
    for rec in out:
        for mode in ("V1", "V2"):
            r = rec[mode]
            ser = r["series"].to_numpy(dtype=np.float32)
            h = hashlib.sha1((json.dumps(rec["cand"]["spec"], sort_keys=True) + "|a8val|" + mode).encode()).hexdigest()
            rows.append(dict(stage="a8val", tf=rec["cand"]["combo"]["tf"], spec_json=json.dumps(rec["cand"]["spec"]), spec_hash=h,
                             parent_hash=None, family=f"a8val:{mode}", tag=rec["cand"]["name"], fold_json=json.dumps(r["per_coin"]),
                             median_oos_sharpe=r["med_sharpe"], mean_is_sharpe=None, n_pos_folds=int(r["frac_pos"] * r["n_coins"]),
                             is_trades=None, min_oos_trades=None, oos_sharpe=r["ew_sharpe"], max_dd=r["ew_dd"],
                             profit_factor=min(r["pf"], 99.0), oos_returns=ser.tobytes()))
    st.insert_many(rows)
    N = st.count()
    st.close()
    for mode in ("V1", "V2"):
        ts = []
        for rec in out:
            x = rec[mode]["series"].to_numpy()
            if x.std(ddof=1) > 0:
                ts.append(x.mean() / x.std(ddof=1))
        for rec in out:
            x = rec[mode]["series"].to_numpy()
            d = deflated_sharpe_ratio(x, np.array(ts), n_trials=N, periods_per_year=PPY)
            rec[mode]["dsr"] = float(d.dsr)
    for rec in out:
        rec["N"] = N

    def f(x, pct=False):
        return "n/a" if x is None or not np.isfinite(x) else (f"{x*100:.1f}%" if pct else f"{x:.2f}")

    passed = [r for r in out if r["pass"]]
    L = [f"# A8 validation (V1 = 25 fresh coins, V2 = 8 design coins after 2026-03-10 14:00 UTC)", ""]
    L.append(f"**VERDICT: {len(passed)} of {len(out)} candidates pass V1 and V2**"
             + (": " + ", ".join(r['cand']['name'] for r in passed) if passed else " (nothing validated; the top-3 by V1 score are marked 'nao validada')"))
    for rec in out:
        n = rec["cand"]["name"]
        L.append(f"- {n} ({fam_s(rec['cand'])}): V1 {'PASS' if rec['V1']['verdict']['ok'] else 'FAIL'}, V2 {'PASS' if rec['V2']['verdict']['ok'] else 'FAIL'} "
                 f"=> **{'VALIDADA' if rec['pass'] else 'REJEITADA'}**")
    L += ["", f"DSR deflated at N = {out[0]['N']} (all rows of the trials table incl. a8 design, T3 and these validation rows); trial-Sharpe variance from the 10 candidates on that set. "
          "Rules (A8): median-over-coins Sharpe > 0.5; Sharpe > 0 on >= 60% of coins; pooled profit factor > 1.15; EW portfolio maxDD <= 0.7 x EW B&H maxDD; "
          "on V1 also EW portfolio Sharpe >= EW B&H Sharpe. V1 starts 120 days after each coin's first bar; V2 evaluates bars >= cutoff with continuous signal state. "
          "Costs BINANCE_PERP_TAKER + funding; fills next-bar open. B&H = perp close-to-close incl. funding, equal weight, daily rebalance.", "",
          "## Benchmarks", "", "| set | EW B&H Sharpe | EW B&H maxDD | EW B&H CAGR | brayvb_v2 default 4h EW Sharpe | brayvb maxDD | brayvb CAGR | brayvb PF | brayvb med coin Sharpe |", "|" + "---|" * 9]
    for m in ("V1", "V2"):
        b = bray[m]
        L.append(f"| {m} ({b['n_coins']} coins) | {f(b['bh_sharpe'])} | {f(b['bh_dd'], True)} | {f(b['bh_cagr'], True)} | {f(b['ew_sharpe'])} | {f(b['ew_dd'], True)} | {f(b['ew_cagr'], True)} | {f(b['pf'])} | {f(b['med_sharpe'])} |")
    for m in ("V1", "V2"):
        L += ["", f"## {m} per candidate", "",
              "| cand | combo | med coin Sharpe | % coins>0 | PF | EW Sharpe | EW maxDD | EW CAGR | EW B&H Sharpe | B&H maxDD | short pnl | long pnl | trades | DSR p | pass |", "|" + "---|" * 15]
        for rec in out:
            r = rec[m]
            L.append(f"| {rec['cand']['name']} | {fam_s(rec['cand'])} | {f(r['med_sharpe'])} | {f(r['frac_pos'], True)} | {f(r['pf'])} | {f(r['ew_sharpe'])} | {f(r['ew_dd'], True)} | "
                     f"{f(r['ew_cagr'], True)} | {f(r['bh_sharpe'])} | {f(r['bh_dd'], True)} | {f(r['short_pnl'])} | {f(r['long_pnl'])} | {r['trades']} | {f(r['dsr'])} | {'Y' if r['verdict']['ok'] else 'N'} |")
    L += ["", "## Best vs brayvb_v2 and EW B&H", ""]
    best = max(out, key=lambda r: (r["pass"], r["V1"]["ew_sharpe"]))
    for m in ("V1", "V2"):
        r, b = best[m], bray[m]
        L.append(f"- {m}: {best['cand']['name']} EW Sharpe {f(r['ew_sharpe'])} / maxDD {f(r['ew_dd'], True)} vs brayvb {f(b['ew_sharpe'])} / {f(b['ew_dd'], True)} vs EW B&H {f(r['bh_sharpe'])} / {f(r['bh_dd'], True)}")
    L.append(f"\nRuntime of validation: {time.time()-t0:.0f}s.")
    (ROOT / "research/reports/a8_validation.md").write_text("\n".join(L), encoding="utf-8")
    slim = [dict(name=r["cand"]["name"], pass_=r["pass"], V1={k: v for k, v in r["V1"].items() if k not in ("series", "per_coin")},
                 V2={k: v for k, v in r["V2"].items() if k not in ("series", "per_coin")}) for r in out]
    (ROOT / "research/runs/a8_validation.json").write_text(json.dumps(slim, indent=1, default=float))
    print("\n".join(L[:14]))


def fam_s(c):
    cb = c["combo"]
    return f"{cb['template']}:{cb['filter']}:{cb['trigger']}:{cb['tf']}:{cb['source']}"


if __name__ == "__main__":
    main()
