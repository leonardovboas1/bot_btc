"""H7 evaluation exactly as pre-registered (docs/preregistration.md, Amendment 3).

Run:  python research/h7_gate.py [--placebos 20] [--no-db]
Never loads holdout data (all loaders get end=cutoff; folds end at the cutoff).

Operational decisions fixed BEFORE the first run (also written at the top of the report):
 D1  "realistic" cost preset = btclab.engine.costs.BINANCE_PERP_TAKER (0.05%/side + 2 bp slippage/side);
     "2x costs" doubles fee and slippage (funding is not doubled).
 D2  Class balance: TP-first is 10-27 % of events, so a literal p>0.52 gate would almost never fire. Both
     gates are trained with balanced class weights so that p=0.5 is the no-information point the sizing
     rule (p-0.5)/0.25 assumes. Logit C=1.0 (not in the grid, fixed). Precision = share of taken trades whose
     label is y=1 (TP first), as in the label definition; win rate (net PnL>0) is reported as information.
 D3  Model inputs = the builder's FEATURE_COLS + the 4 coin one-hots + event descriptors (direction, 4 trigger
     one-hots); direction/trigger are part of the event definition, without them the label is undefined.
     NaN -> 0 after standardisation for the logit (A1.4 "no signal"); HistGradientBoosting handles NaN natively.
 D4  Folds: walk_forward_folds on the BTC feature index (6 folds, 18m train / 6m test, end = holdout cutoff,
     embargo 24h, purge_bars=0).  Purge by label: a training event is kept only if bar_time >= train_start and
     end_h + 24h <= test_start - 24h (embargo).  Test events: bar_time in [test_start, test_end].
 D5  One open position per coin; events that arrive while a position in the same coin is open are skipped.
     Same-bar ties: gated -> highest p first, ungated -> trigger order taker, basis, oi, liq.  Ungated = all
     events, size 1, same h, same rules.
 D6  Sizing: notional = size * R0 / (ATR/entry), R0 = 0.2 % of equity risked per size-1 trade at the 1-ATR stop
     (equity base fixed at 1, non-compounded); if summed gross notional of open trades exceeds 1, every trade
     in that hour is scaled down proportionally (computed per trade at its worst hour).
 D7  Costs per trade = notional * 2 * (fee + slippage); funding = direction * notional * sum of 8h prints
     settled in (entry, exit].  Trade PnL is booked at the exit hour; Sharpe from daily sums (UTC), 365/yr.
 D8  (i) gated portfolio fold Sharpe > ungated (same h) in >= 4/6 folds; (iii) GBT beats logit in >= 7 of the
     12 matched (h, p) pairs; if (iii) fails the logistic gate is the candidate; (iv) >= 20 label shuffles of the
     training set, mean placebo Sharpe gain >= 80 % of the real gain (real gain <= 0 also fails);
     (vi) permutation importance = mean OOS AUC drop of a jointly permuted family block (5 repeats, per fold),
     share among the 5 signal families (context block reported separately); fails if max share > 60 %.
 D9  Candidate = best pooled OOS Sharpe of the 24 configs (selection is paid for in DSR N).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from btclab.data import loaders
from btclab.engine.costs import BINANCE_PERP_TAKER
from btclab.features.h7 import FEATURE_COLS, SYMBOLS
from btclab.research.stats.cscv import cscv_pbo
from btclab.research.stats.dsr import deflated_sharpe_ratio, sharpe_ratio
from btclab.research.stats.walkforward import walk_forward_folds

FEAT_DIR = ROOT / "data" / "features"
REPORT = ROOT / "research" / "reports" / "h7_results.md"
PNG = ROOT / "research" / "reports" / "h7_equity.png"
HOUR = 3_600_000_000_000
HS = (2, 4, 6)
PMINS = (0.52, 0.55, 0.58, 0.60)
GATES = ("logit", "gbt")
R0 = 0.002
COST = BINANCE_PERP_TAKER
TRIG = ("taker", "basis", "oi", "liq")
COINS = [s.replace("USDT", "") for s in SYMBOLS]
COIN_COLS = [f"coin_{c}" for c in COINS]
COLS = FEATURE_COLS + ["direction"] + [f"trig_{t}" for t in TRIG] + COIN_COLS
FAMILIES = {
    "order_flow": ["taker_z24", "taker_z168", "taker_burst"],
    "liquidation_map": ["liq_long_dist_atr", "liq_short_dist_atr", "liq_long_size", "liq_short_size",
                        "liq_long_consumed", "liq_short_consumed"],
    "positioning": ["doi1_z", "doi4_z", "doi24_z", "funding_z30d", "ls_global_z", "ls_top_z"],
    "basis": ["basis_z7d", "basis_chg4h"],
    "session": ["hour_sin", "hour_cos", "wd_sin", "wd_cos", "sess_asia", "sess_europe", "sess_us"],
}
FAMILIES["_context"] = ["rvol_z", "ret_24h", "er24", "direction"] + [f"trig_{t}" for t in TRIG] + COIN_COLS
assert sorted(sum(FAMILIES.values(), [])) == sorted(COLS)
FAM_IDX = {k: [COLS.index(c) for c in v] for k, v in FAMILIES.items()}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data
def ns(x) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(x, utc=True)).as_unit("ns").asi8


def load_all(cutoff):
    frames, funds = [], {}
    for k, s in enumerate(SYMBOLS):
        f = pd.read_parquet(FEAT_DIR / f"h7_{s}_1h.parquet")
        e = pd.read_parquet(FEAT_DIR / f"h7_{s}_1h_events.parquet")
        assert f.index.max() < cutoff and e["bar_time"].max() < cutoff
        x = e.merge(f[[c for c in FEATURE_COLS + COIN_COLS]], left_on="bar_time", right_index=True, how="left")
        x["sym"] = k
        frames.append(x)
        fu = loaders.load_funding(end=cutoff, symbol=s)
        funds[k] = (ns(fu.index), fu.to_numpy(float))
    ev = pd.concat(frames, ignore_index=True)
    for t in TRIG:
        ev[f"trig_{t}"] = (ev["trigger"] == t).astype(float)
    ev["trig_rank"] = ev["trigger"].map({t: i for i, t in enumerate(TRIG)})
    ev["bar_ns"] = ns(ev["bar_time"])
    ev["entry_ns"] = ns(ev["entry_time"])
    return ev, funds


def prep_h(ev, funds, h, folds):
    """Event table for horizon h with fold assignment (-1 = not test) and per-unit-notional net outcome."""
    d = ev[ev[f"y_{h}"].notna()].copy().reset_index(drop=True)
    d["end_ns"] = ns(d[f"end_{h}"])
    d["y"] = d[f"y_{h}"].to_numpy()
    atr_pct = d["atr"] / d["entry_price"]
    d["atr_pct"] = atr_pct
    d["gross"] = d[f"ret_atr_{h}"] * atr_pct                       # price return of the trade (direction applied)
    fund = np.zeros(len(d))
    for k, (fi, fr) in funds.items():
        m = (d["sym"] == k).to_numpy()
        cs = np.concatenate([[0.0], np.cumsum(fr)])
        a = np.searchsorted(fi, d.loc[m, "entry_ns"].to_numpy(), side="right")   # prints with t > entry
        b = np.searchsorted(fi, d.loc[m, "end_ns"].to_numpy(), side="right")     # prints with t <= end
        fund[m] = d.loc[m, "direction"].to_numpy() * (cs[b] - cs[a])
    d["fund"] = fund
    fold = np.full(len(d), -1)
    for f in folds:
        m = (d["bar_ns"] >= f.test_start.value) & (d["bar_ns"] <= f.test_end.value)
        fold[m.to_numpy()] = f.fold
    d["fold"] = fold
    return d


# ----------------------------------------------------------------------------- models
def make_model(gate, seed=0):
    if gate == "logit":
        return LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=500)
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, early_stopping=False,
                                          class_weight="balanced", random_state=seed)


def standardize(Xtr, Xte):
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    f = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0)
    return f(Xtr), f(Xte)


def fit_predict(gate, Xtr, ytr, Xte, seed=0):
    if gate == "logit":
        a, b = standardize(Xtr, Xte)
    else:
        a, b = Xtr, Xte
    m = make_model(gate, seed).fit(a, ytr)
    return m, m.predict_proba(b)[:, 1], b


def train_idx(d, fold):
    ts, te = fold.test_start.value, fold.train_start.value
    ok = (d["bar_ns"] >= te) & (d["end_ns"] + 24 * HOUR <= ts - 24 * HOUR)
    return np.flatnonzero(ok.to_numpy())


# ----------------------------------------------------------------------------- simulation
class Grid:
    def __init__(self, folds):
        self.t0 = (folds[0].test_start.value // HOUR) * HOUR
        t1 = folds[-1].test_end.value // HOUR * HOUR + HOUR
        self.n = int((t1 - self.t0) // HOUR) + 1
        self.hours = pd.to_datetime(self.t0 + np.arange(self.n) * HOUR, utc=True)
        self.folds = folds
        self.fold_slices = [(int((f.test_start.value // HOUR * HOUR - self.t0) // HOUR),
                             int((f.test_end.value // HOUR * HOUR - self.t0) // HOUR) + 1) for f in folds]
        self.oos = np.zeros(self.n, bool)
        for a, b in self.fold_slices:
            self.oos[a:b] = True


def select_trades(d, p, pmin):
    """Chronological one-position-per-coin walk. p=None -> ungated. Returns index array of taken events."""
    m = (d["fold"].to_numpy() >= 0)
    if p is not None:
        m &= np.nan_to_num(p, nan=0.0) > pmin
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return idx
    sym, ent, end = d["sym"].to_numpy()[idx], d["entry_ns"].to_numpy()[idx], d["end_ns"].to_numpy()[idx]
    tie = -p[idx] if p is not None else d["trig_rank"].to_numpy()[idx].astype(float)
    order = np.lexsort((tie, ent, sym))
    taken, last_sym, busy = [], -1, -1
    for o in order:
        s = sym[o]
        if s != last_sym:
            last_sym, busy = s, -1
        if ent[o] >= busy:
            taken.append(idx[o])
            busy = end[o]
    return np.array(taken, dtype=int)


def simulate(d, grid, taken, p, cost_mult=1.0):
    """Portfolio simulation of the taken events. Returns (hourly pnl array, per-trade DataFrame)."""
    if taken.size == 0:
        return np.zeros(grid.n), pd.DataFrame(columns=["sym", "fold", "pnl", "y", "gross", "size"])
    t = d.iloc[taken]
    size = np.ones(len(t)) if p is None else np.minimum(1.0, (p[taken] - 0.5) / 0.25)
    notional = size * R0 / t["atr_pct"].to_numpy()
    e_h = ((t["entry_ns"].to_numpy() - grid.t0) // HOUR).astype(int)
    x_h = np.maximum(((t["end_ns"].to_numpy() - grid.t0) // HOUR).astype(int) - 1, e_h)   # last hour held (end is bar close)
    diff = np.zeros(grid.n + 2)
    np.add.at(diff, e_h, notional)
    np.add.at(diff, x_h + 1, -notional)
    gross = np.cumsum(diff)[: grid.n]
    peak = np.array([gross[a:b + 1].max() for a, b in zip(e_h, x_h)])
    scale = np.minimum(1.0, 1.0 / np.maximum(peak, 1e-12))
    nu = notional * scale
    cost = 2.0 * (COST.fee_rate + COST.slippage_bp * 1e-4) * cost_mult
    pnl = nu * (t["gross"].to_numpy() - t["fund"].to_numpy() - cost)
    hourly = np.zeros(grid.n)
    np.add.at(hourly, x_h, pnl)
    tr = pd.DataFrame({"sym": t["sym"].to_numpy(), "fold": t["fold"].to_numpy(), "pnl": pnl,
                       "y": t["y"].to_numpy(), "gross": t["gross"].to_numpy(), "size": size, "scale": scale})
    return hourly, tr


def daily(hourly, grid, sl=None):
    s = pd.Series(hourly, index=grid.hours)
    if sl is not None:
        s = s.iloc[sl[0]:sl[1]]
    return s.resample("1D").sum()


def sh(x):
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x, ddof=1) == 0:
        return 0.0
    return float(sharpe_ratio(x, 365.0))


def oos_daily(hourly, grid):
    return pd.Series(hourly[grid.oos], index=grid.hours[grid.oos]).resample("1D").sum()


def evaluate(d, grid, p, pmin):
    taken = select_trades(d, p, pmin)
    out = {}
    for cm in (1.0, 2.0):
        hourly, tr = simulate(d, grid, taken, p, cm)
        od = oos_daily(hourly, grid)
        out[cm] = dict(hourly=hourly, tr=tr, od=od, sharpe=sh(od.to_numpy()),
                       fold_sh=[sh(daily(hourly, grid, sl).to_numpy()) for sl in grid.fold_slices])
    tr = out[1.0]["tr"]
    out["n"] = len(tr)
    out["prec"] = float(tr["y"].mean()) if len(tr) else np.nan
    out["win"] = float((tr["pnl"] > 0).mean()) if len(tr) else np.nan
    out["taken"] = taken
    return out


def max_dd(od):
    eq = 1.0 + np.cumsum(np.asarray(od, float))
    return float((np.maximum.accumulate(eq) - eq).max())


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebos", type=int, default=20)
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args(argv)
    cutoff = loaders.get_cutoff()
    log("cutoff", cutoff)
    ev, funds = load_all(cutoff)
    btc_idx = pd.read_parquet(FEAT_DIR / "h7_BTCUSDT_1h.parquet", columns=["close"]).index
    folds = walk_forward_folds(btc_idx, n_folds=6, train_months=18, test_months=6, purge_bars=0,
                               embargo="24h", end=cutoff)
    for f in folds:
        log(f)
    grid = Grid(folds)

    data, preds, models, tests = {}, {}, {}, {}
    ungated = {}
    for h in HS:
        d = prep_h(ev, funds, h, folds)
        data[h] = d
        X = d[COLS].to_numpy(float)
        y = d["y"].to_numpy()
        ungated[h] = evaluate(d, grid, None, None)
        for g in GATES:
            p = np.full(len(d), np.nan)
            for f in folds:
                tr_i = train_idx(d, f)
                te_i = np.flatnonzero((d["fold"] == f.fold).to_numpy())
                m, pt, b = fit_predict(g, X[tr_i], y[tr_i], X[te_i])
                p[te_i] = pt
                models[(g, h, f.fold)] = m
                tests[(g, h, f.fold)] = (te_i, b)
                log(f"h={h} {g} fold {f.fold}: train {len(tr_i)} test {len(te_i)} AUC "
                    f"{roc_auc_score(y[te_i], pt):.3f} mean p {pt.mean():.3f}")
            preds[(g, h)] = p

    # ---- 24 configs
    res = {}
    for h in HS:
        for g in GATES:
            for pm in PMINS:
                res[(g, h, pm)] = evaluate(data[h], grid, preds[(g, h)], pm)
    keys = list(res)
    rows = []
    for k in keys:
        g, h, pm = k
        r, u = res[k], ungated[h]
        beats = int(sum(a_ > b_ for a_, b_ in zip(r[1.0]["fold_sh"], u[1.0]["fold_sh"])))
        rows.append(dict(gate=g, h=h, p_min=pm, trades=r["n"], sharpe=r[1.0]["sharpe"], ungated=u[1.0]["sharpe"],
                         beat_folds=beats, precision=r["prec"], win=r["win"], sharpe_2x=r[2.0]["sharpe"],
                         maxdd=max_dd(r[1.0]["od"]), ret=float(r[1.0]["od"].sum()),
                         pos_folds=int(sum(s > 0 for s in r[1.0]["fold_sh"]))))
    tab = pd.DataFrame(rows)
    best = tab.sort_values("sharpe", ascending=False).iloc[0]
    log("best", best.to_dict())

    # ---- (iii) GBT vs logit on matched pairs
    pair = tab.pivot_table(index=["h", "p_min"], columns="gate", values="sharpe")
    gbt_wins = int((pair["gbt"] > pair["logit"]).sum())
    iii_pass = gbt_wins >= 7
    best_of = {g: tab[tab.gate == g].sort_values("sharpe", ascending=False).iloc[0] for g in GATES}
    cand = best if (best["gate"] == "logit" or iii_pass) else best_of["logit"]
    ck = (cand["gate"], int(cand["h"]), float(cand["p_min"]))
    cr, cu = res[ck], ungated[ck[1]]
    log("candidate", ck)

    # ---- (iv) placebo
    def placebo(key):
        g, h, pm = key
        d, X, y = data[h], data[h][COLS].to_numpy(float), data[h]["y"].to_numpy()
        real_gain = res[key][1.0]["sharpe"] - ungated[h][1.0]["sharpe"]
        sh_list = []
        for s in range(a.placebos):
            rng = np.random.default_rng(1000 + s)
            p = np.full(len(d), np.nan)
            for f in folds:
                tr_i = train_idx(d, f)
                te_i = tests[(g, h, f.fold)][0]
                _, pt, _ = fit_predict(g, X[tr_i], rng.permutation(y[tr_i]), X[te_i], seed=s)
                p[te_i] = pt
            sh_list.append(evaluate(d, grid, p, pm)[1.0]["sharpe"])
        gains = np.array(sh_list) - ungated[h][1.0]["sharpe"]
        return dict(real_gain=real_gain, gains=gains, mean_gain=float(gains.mean()),
                    frac_ge=float((gains >= 0.8 * real_gain).mean()),
                    fail=bool(real_gain <= 0 or gains.mean() >= 0.8 * real_gain))
    plc = {}
    for key in {ck, (best_of["logit"]["gate"], int(best_of["logit"]["h"]), float(best_of["logit"]["p_min"])),
                (best_of["gbt"]["gate"], int(best_of["gbt"]["h"]), float(best_of["gbt"]["p_min"]))}:
        log("placebo", key)
        plc[key] = placebo(key)

    # ---- (vi) permutation importance
    def perm_importance(g, h, reps=5):
        d, y = data[h], data[h]["y"].to_numpy()
        imp = {k: [] for k in FAM_IDX}
        for f in folds:
            te_i, b = tests[(g, h, f.fold)]
            m = models[(g, h, f.fold)]
            yt = y[te_i]
            base = roc_auc_score(yt, m.predict_proba(b)[:, 1])
            rng = np.random.default_rng(7 + f.fold)
            for fam, ix in FAM_IDX.items():
                drops = []
                for _ in range(reps):
                    bb = b.copy()
                    perm = rng.permutation(len(bb))
                    bb[:, ix] = b[perm][:, ix]
                    drops.append(base - roc_auc_score(yt, m.predict_proba(bb)[:, 1]))
                imp[fam].append(np.mean(drops))
        return {k: float(np.mean(v)) for k, v in imp.items()}
    imps = {}
    for key in {(ck[0], ck[1]), (best_of["logit"]["gate"], int(best_of["logit"]["h"])),
                (best_of["gbt"]["gate"], int(best_of["gbt"]["h"]))}:
        log("perm importance", key)
        imps[key] = perm_importance(*key)

    # ---- DSR / PBO
    mat = pd.DataFrame({f"{g}|h{h}|p{pm}": res[(g, h, pm)][1.0]["od"] for (g, h, pm) in keys})
    n_h7 = len(keys)
    od_c = cr[1.0]["od"].to_numpy()
    T = len(od_c)
    store_n = None
    if not a.no_db:
        store_n = write_trials(keys, res, tab, grid)
    n_total = store_n if store_n else n_h7
    srs = np.array([sharpe_ratio(mat[c].to_numpy()) for c in mat.columns])
    dsr_fam = deflated_sharpe_ratio(od_c, n_trials=n_h7, var_trial_sharpes=float(np.nanvar(srs, ddof=1)), periods_per_year=365)
    dsr_tot_null = deflated_sharpe_ratio(od_c, n_trials=n_total, var_trial_sharpes=1.0 / max(T, 2), periods_per_year=365)
    dsr_tot_emp = deflated_sharpe_ratio(od_c, n_trials=n_total, var_trial_sharpes=float(np.nanvar(srs, ddof=1)), periods_per_year=365)
    pbo = cscv_pbo(mat.fillna(0.0), n_blocks=16)

    # ---- per coin and benchmark
    percoin = per_coin(data[ck[1]], grid, cr, cu)
    perp = loaders.load_ohlcv("perp", "1h", end=cutoff, symbol="BTCUSDT")["close"]
    bh = perp.resample("1D").last().pct_change()
    bh = bh[(bh.index >= pd.Timestamp(grid.hours[0]).normalize()) & (bh.index <= pd.Timestamp(grid.hours[-1]).normalize())]
    bh_oos = bh.reindex(cr[1.0]["od"].index).fillna(0.0)

    write_report(tab, best, cand, ck, res, ungated, pair, gbt_wins, iii_pass, plc, imps, dsr_fam, dsr_tot_null,
                 dsr_tot_emp, pbo, n_total, n_h7, percoin, bh_oos, folds, grid, a.placebos, best_of, data)
    log("done")


def per_coin(d, grid, cr, cu):
    out = {}
    for name, run in (("gated", cr), ("ungated", cu)):
        t = d.iloc[run["taken"]]
        tr = run[1.0]["tr"]
        end_h = np.maximum(((t["end_ns"].to_numpy() - grid.t0) // HOUR).astype(int) - 1,
                           ((t["entry_ns"].to_numpy() - grid.t0) // HOUR).astype(int))
        out[name] = {}
        for k, c in enumerate(COINS):
            m = (tr["sym"].to_numpy() == k)
            hourly = np.zeros(grid.n)
            np.add.at(hourly, end_h[m], tr["pnl"].to_numpy()[m])
            od = oos_daily(hourly, grid)
            out[name][c] = (sh(od.to_numpy()), float(od.sum()), int(m.sum()),
                            float(tr["y"].to_numpy()[m].mean()) if m.any() else float("nan"))
    return out


def write_trials(keys, res, tab, grid):
    import duckdb  # noqa
    from btclab.research.search.trials import TrialStore
    store = TrialStore()
    store.con.execute("DELETE FROM trials WHERE stage='h7'")
    recs = []
    for (g, h, pm), row in zip(keys, tab.to_dict("records")):
        r = res[(g, h, pm)]
        spec = dict(family="h7", gate=g, h=h, p_min=pm, costs="BINANCE_PERP_TAKER", amendment=3)
        sj = json.dumps(spec, sort_keys=True)
        od = r[1.0]["od"].to_numpy()
        fsh = r[1.0]["fold_sh"]
        trn = r[1.0]["tr"]
        w, l = trn.loc[trn.pnl > 0, "pnl"].sum(), -trn.loc[trn.pnl < 0, "pnl"].sum()
        recs.append(dict(stage="h7", tf="1h", spec_json=sj, spec_hash=hashlib.sha1(sj.encode()).hexdigest(),
                         parent_hash=None, family=f"h7:{g}:h{h}:p{pm}", tag="amendment3",
                         fold_json=json.dumps([dict(fold=i, oos_sharpe=s) for i, s in enumerate(fsh)]),
                         median_oos_sharpe=float(np.median(fsh)), mean_is_sharpe=None,
                         n_pos_folds=int(sum(s > 0 for s in fsh)), is_trades=None, min_oos_trades=int(
                             min((trn["fold"] == i).sum() for i in range(6))) if len(trn) else 0,
                         oos_sharpe=r[1.0]["sharpe"], max_dd=max_dd(od), profit_factor=float(min(w / l, 99.0)) if l > 0 else 0.0,
                         oos_returns=od.astype(np.float32).tobytes()))
    store.insert_many(recs)
    n = store.count()
    store.close()
    log("trials table N =", n)
    return n


def write_report(tab, best, cand, ck, res, ungated, pair, gbt_wins, iii_pass, plc, imps, dsr_fam, dsr_null, dsr_emp,
                 pbo, n_total, n_h7, percoin, bh_oos, folds, grid, n_shuf, best_of, data):
    g, h, pm = ck
    cr, cu = res[ck], ungated[h]
    # tests on candidate
    beats = int(sum(x > y for x, y in zip(cr[1.0]["fold_sh"], cu[1.0]["fold_sh"])))
    t1 = beats >= 4
    t2 = bool(cr["prec"] > 0.53)
    t3 = iii_pass
    pl = plc[ck]
    t4 = not pl["fail"]
    t5 = cr[2.0]["sharpe"] > 0
    imp = imps[(g, h)]
    fam5 = {k: max(v, 0.0) for k, v in imp.items() if k != "_context"}
    tot = sum(fam5.values())
    shares = {k: (v / tot if tot > 0 else float("nan")) for k, v in fam5.items()}
    top = max(shares, key=lambda k: shares[k] if np.isfinite(shares[k]) else -1)
    t6 = bool(np.isfinite(shares[top]) and shares[top] <= 0.60)
    verdicts = [("(i) OOS Sharpe beats ungated in >= 4/6 folds", t1, f"{beats}/6 folds"),
                ("(ii) OOS precision of taken trades > 53%", t2, f"{cr['prec']:.1%} on {cr['n']} trades"),
                ("(iii) GBT beats logit OOS", t3, f"GBT better in {gbt_wins}/12 matched (h,p) pairs"
                 + ("" if t3 else "; only the logistic variant may proceed")),
                ("(iv) shuffled-label placebo gain < 80% of real gain", t4,
                 f"real gain {pl['real_gain']:+.2f}, placebo mean gain {pl['mean_gain']:+.2f} ({n_shuf} shuffles)"),
                ("(v) Sharpe at 2x costs > 0", t5, f"{cr[2.0]['sharpe']:+.2f}"),
                ("(vi) no family > 60% of permutation importance", t6,
                 f"max {top} {shares[top]:.0%}" if np.isfinite(shares[top]) else "no positive importance")]
    allpass = all(v[1] for v in verdicts)
    L = []
    L.append(f"**H7 {'PASSED all six falsification tests' if allpass else 'REJECTED'}** (candidate {g} h={h} p_min={pm}, "
             "portfolio, OOS only): " + "; ".join(f"{n.split(' ')[0]} {'PASS' if ok else 'REJECTED'}" for n, ok, _ in verdicts) + ".")
    L += ["", "# H7 results (Amendment 3, walk-forward OOS only, holdout never loaded)", "",
          "## Decisions fixed before the first run", ""]
    doc = __doc__.split("Operational decisions fixed BEFORE the first run (also written at the top of the report):\n")[1]
    L += ["```", doc.strip(), "```", "",
          "Not specified by the amendment and resolved above: the 'realistic' preset does not exist by that name in "
          "`btclab/engine/costs.py`, `BINANCE_PERP_TAKER` (the project default) was used. Folds: alt coins start 2022-01-01, so "
          "fold 0 trains on fewer alt events than BTC.", "",
          "## Falsification tests (candidate, portfolio)", "", "| test | verdict | evidence |", "|---|---|---|"]
    for n, ok, ev_ in verdicts:
        L.append(f"| {n} | **{'PASS' if ok else 'REJECTED'}** | {ev_} |")
    L += ["", f"Best of the 24 configs by pooled OOS Sharpe: {best['gate']} h={int(best['h'])} p_min={best['p_min']} "
          f"(Sharpe {best['sharpe']:.2f}). Candidate after (iii): {g} h={h} p_min={pm}.", "",
          f"Candidate OOS: Sharpe {cr[1.0]['sharpe']:.2f} vs ungated (h={h}) {cu[1.0]['sharpe']:.2f}; precision {cr['prec']:.1%}; "
          f"win rate {cr['win']:.1%}; trades {cr['n']} (ungated {cu['n']}); maxDD {max_dd(cr[1.0]['od']):.3f} equity units "
          f"(ungated {max_dd(cu[1.0]['od']):.3f}).", "",
          "## Statistical adjustment", "",
          f"- DSR, family only (N={n_h7}, variance of the {n_h7} trial Sharpes): DSR={dsr_fam.dsr:.3f}, PSR(0)={dsr_fam.psr:.3f}, "
          f"annualised SR {dsr_fam.sr_annualised:.2f}, passed(>0.95)={dsr_fam.passed}.",
          f"- DSR, N={n_total} (whole trials table incl. these {n_h7}), trial-Sharpe variance 1/T (null, as gauntlet.py): "
          f"DSR={dsr_null.dsr:.3f}, passed={dsr_null.passed}. With the empirical variance of the {n_h7} H7 Sharpes: DSR={dsr_emp.dsr:.3f}.",
          f"- PBO (CSCV, 16 blocks, {pbo.n_trials} configs, {pbo.n_splits} splits): **{pbo.pbo:.3f}** (gate < 0.15 / 0.20); "
          f"mean OOS Sharpe of IS-best {pbo.mean_oos_best:.3f}, P(loss) {pbo.prob_loss:.2f}.", "",
          "## All 24 configurations (portfolio, OOS pooled, 1x costs unless stated)", "",
          "| gate | h | p_min | trades | Sharpe | ungated Sharpe (h) | folds > ungated | folds Sharpe>0 | precision | win rate | Sharpe 2x costs | maxDD | OOS net return |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        L.append(f"| {r.gate} | {r.h} | {r.p_min} | {r.trades} | {r.sharpe:.2f} | {r.ungated:.2f} | {r.beat_folds}/6 | {r.pos_folds}/6 | "
                 f"{r.precision:.1%} | {r.win:.1%} | {r.sharpe_2x:.2f} | {r.maxdd:.3f} | {r.ret:+.3f} |")
    L += ["", "## GBT vs logit matched pairs (OOS Sharpe)", "", "| h | p_min | logit | GBT |", "|---|---|---|---|"]
    for (hh, pp), r in pair.iterrows():
        L.append(f"| {hh} | {pp} | {r['logit']:.2f} | {r['gbt']:.2f} |")
    L += ["", "## Per-fold OOS Sharpe (candidate vs ungated)", "", "| fold | test window | gated | ungated |", "|---|---|---|---|"]
    for f, a_, b_ in zip(folds, cr[1.0]["fold_sh"], cu[1.0]["fold_sh"]):
        L.append(f"| {f.fold} | {f.test_start.date()} .. {f.test_end.date()} | {a_:.2f} | {b_:.2f} |")
    L += ["", "## Per coin (candidate vs ungated, OOS, equity base 1 shared across coins)", "",
          "| coin | gated Sharpe | gated net ret | gated trades | gated precision | ungated Sharpe | ungated net ret | ungated trades | ungated precision |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in COINS:
        a_, b_ = percoin["gated"][c], percoin["ungated"][c]
        L.append(f"| {c} | {a_[0]:.2f} | {a_[1]:+.3f} | {a_[2]} | {a_[3]:.1%} | {b_[0]:.2f} | {b_[1]:+.3f} | {b_[2]} | {b_[3]:.1%} |")
    L += ["", "## Placebo gates (shuffled training labels)", "", "| config | real gain | placebo mean gain | share of shuffles >= 80% of real gain | verdict |", "|---|---|---|---|---|"]
    for k, v in plc.items():
        L.append(f"| {k[0]} h={k[1]} p={k[2]} | {v['real_gain']:+.2f} | {v['mean_gain']:+.2f} | {v['frac_ge']:.0%} | {'REJECTED' if v['fail'] else 'PASS'} |")
    L += ["", "## Permutation importance (mean OOS AUC drop per family block)", "",
          "| gate h | " + " | ".join(FAM_IDX) + " | max signal-family share |", "|---|" + "---|" * (len(FAM_IDX) + 1)]
    for k, v in imps.items():
        f5 = {kk: max(vv, 0) for kk, vv in v.items() if kk != "_context"}
        s_ = sum(f5.values())
        mx = max(f5.values()) / s_ if s_ > 0 else float("nan")
        L.append(f"| {k[0]} h={k[1]} | " + " | ".join(f"{v[n]:+.4f}" for n in FAM_IDX) + f" | {mx:.0%} |")
    # equity
    od_g, od_u = cr[1.0]["od"], cu[1.0]["od"]
    stats = []
    for name, s in (("H7 gated (candidate)", od_g), ("Ungated primary", od_u), ("BTC buy & hold", bh_oos)):
        eq = 1 + s.cumsum() if "hold" not in name else (1 + s).cumprod()
        stats.append((name, sh(s.to_numpy()), float(eq.iloc[-1] - 1), float(s.std() * np.sqrt(365)), float((eq.cummax() - eq).max())))
    L += ["", "## Equity, OOS only", "", "| series | Sharpe | OOS return | ann. vol | maxDD (equity units) |", "|---|---|---|---|---|"]
    for n, s_, r_, v_, dd in stats:
        L.append(f"| {n} | {s_:.2f} | {r_:+.1%} | {v_:.1%} | {dd:.3f} |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(od_g.index, 1 + od_g.cumsum(), label=f"H7 gated ({g} h={h} p={pm})")
        ax.plot(od_u.index, 1 + od_u.cumsum(), label=f"ungated primary (h={h})")
        ax.plot(bh_oos.index, (1 + bh_oos).cumprod(), label="BTC buy & hold", color="grey")
        ax.set_title("OOS equity (strategies: R0=0.2% risk per trade, non-compounded; gaps between folds are not removed)")
        ax.title.set_fontsize(8)
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(PNG, dpi=110); plt.close(fig)
        L += ["", f"![equity]({PNG.name})", ""]
    except Exception as e:  # pragma: no cover
        L += ["", f"(plot failed: {e})", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    log("report written", REPORT)


if __name__ == "__main__":
    main()
