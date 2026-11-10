"""Evaluation of the pre-registered hypotheses H1-H6 (docs/preregistration.md: S1 H1-H4, Amendment 1 H5-H6,
A1.1 depth window, A1.3 trial accounting, A1.4 causal rules, A2 trade minimums, S2, S4).

Run:  python research/h1_h6.py [--no-db]
Never loads holdout data. All daily data are cut at the cutoff DAY (2026-03-10 00:00 UTC): the existing 1d
context keeps the 2026-03-10 daily bar, whose close lies after the 14:00 cutoff, so that bar is dropped here.

OPERATIONAL DECISIONS (fixed BEFORE the first run; repeated at the top of the report)
 D1  Simulator: the engine (btclab/engine/sim.py) only supports discrete entry/exit, not continuous vol-scaled
     positions, so a daily vectorised position simulator is used with the engine's cost model
     BINANCE_PERP_TAKER (fee 0.05% + slippage 2 bp = 7 bp per side on |delta position|) and Binance 8h funding
     (positive rate: longs pay).  Signal on close t, fill at open t+1.  Equity compounding is ignored inside the
     position scale (position in equity units); max |position| = L_max = 2 (S3).  The S3 turnover guard is applied
     to every series: rebalance only if |target - current| > 0.10, always on exit / sign change.  The S3 kill switch,
     soft brake and daily loss limit are NOT applied to hypothesis tests (they are "final model" rules).
 D2  Grids (fixed here; "median parameters" = middle of each registered range):
       H1 sigma-window {20,40,60}, sigma_target {0.2,0.3,0.4}, short_scale {0,0.5,1}, deadband {0,0.125,0.25} (81)
       H1 median = (40, 0.30, 0.5, 0.125) is used as "H1 at median parameters" and as "raw H1" everywhere.
       H2 n {10,25,40}, theta {0.2,0.3,0.4}, z {1.5,2,2.5}, short_scale {0,0.5,1} (81)
       H3 z-window {30,60,90}, tau {1.5,2,2.5,3}, h {0,0.5}, hold {1,3,5,7} (96)
       H4 C {0.01,0.1,1}, p_min {0.50,0.55,0.60}, vertical barrier {5,10,15} (27)
       H5 k {1,3}, w {30,60,90}, tau {1.5,2,2.5,3}, hold {1,3,5,7} (96 per mode; each mode also re-run with
          sum_open_interest_value for the registered value check (96 more each))
       H6 b {1%,2%}, w {20,40,60}, gamma {0.5,1,1.5} (18; placebo 18 more)
 D3  Walk-forward = the standard 6 folds of build_context (18m train / 6m test, 5D embargo, 200-bar purge, end =
     cutoff day).  "WF OOS series" of a hypothesis = per fold the configuration with the best IN-SAMPLE Sharpe on that
     fold's train window (nested selection), stitched over the six test windows.  Raw H1 / B&H / brayvb_v2 are
     compared on the same folds.  B&H = spot-style close-to-close returns (no funding).
 D4  "Trade" = round trip: a maximal run of constant-sign non-zero position (entry from flat or sign flip).  Rebalance
     fills inside a run are not trades (counts of fills are reported as information).  Trade return = run net P&L
     (fees, slippage, funding) / mean |position| of the run.  A2 minimums (40 in-sample trades over the union of train
     windows, 8 per OOS fold) are reported as a flag; S2.1 eligibility = falsification passed + WF + family DSR gate;
     A2 and S3 (>=50 bp net expectancy per round trip, funding bleed <= 15 %/yr) are reported next to it and are
     required for a final model, not for the hypothesis verdict (except H5-standalone, where the 50 bp rule is in the
     registered falsification text).
 D5  DSR: p of the stitched WF OOS series with N = all rows of the trials table (+ these trials + amendments), trial
     Sharpe variance = variance over the hypothesis family matrix (as btclab.research.stats.stages.family_stats);
     also shown: N = family size only, and the all-trials variance.  PBO = CSCV 16 blocks on the family OOS matrix;
     SPA (Hansen) on the family matrix vs B&H.  "DSR-adjusted OOS Sharpe" (H2) = fold Sharpe minus the expected-max
     Sharpe of the family (N = 81, fold-level trial variance), H1 alone has no haircut (N = 1).
 D6  H2: sigma_20 = rolling std of closes (20); RSI(3) Wilder; MR size = 0.5 * sigma_target/sigma_realized of H1
     median; "no position within 2 bars of a regime flip" = flat on the flip bar and the next one; the trend leg and the
     MR leg are two separately guarded books; exit at close >= SMA20 (long) / <= SMA20 (short) or after 5 closes.
     MR expectancy = mean net round-trip return of MR trades in OOS (no MR trades = fails).
 D7  H3: funding z = rolling z of the 8h prints over w days (3w prints, min 0.8), last print <= day end; "close above
     its 5-day low" = close > min of the last 5 daily lows (incl. today); add-on = +0.5 * vol-scaled size while a
     trigger occurred in the last `hold` days; added to H1 (sum).  Untestable if the independent extremes (|z|>tau,
     >=5 d apart) summed over the OOS windows with each fold's selected (w, tau) are < 20.
 D8  H4: primary event = every H1 order that opens or increases exposure (the sign-run entries alone are too few
     to fit 8 features); label y=1 if TP (2 ATR) is hit first or the vertical barrier ends with a positive directional
     return, y=0 otherwise (SL first / same-day both = SL / negative vertical); class-balanced logit so p=0.5 is the
     no-information point; features signed by trade direction (r1, r7, r30, funding z, taker ratio-0.5, Donchian
     distance to the breakout side in ATR); vol z = 20d vol vs trailing 180d.  Training events: t <= test_start - 5d -
     max(10d, V).  Selection of (C, p_min, V) per fold uses in-sample Sharpe with out-of-fold probabilities (3 blocks,
     15-day purge).  Meta size multiplier is set at each event and kept until the run ends or the next event.
     Leak rule: rejected if the mean over folds of the largest |coef| share exceeds 60 %.
 D9  H5: metrics as-of rule of A1.4 (create_time <= daily close - 5 min, max age 12 h, never interpolate); a day is
     "metrics_gap" if any used series is NaN.  Filter mode: fragile longs/shorts halve H1's long/short while the
     condition holds, the flush leg is ADDED (sum) to H1.  Standalone: three legs (fragile longs -> short, fragile shorts
     -> long, flush -> long), 0.5 * vol-scaled size, no pyramiding, OI exit and flush-low stop as written.  Value check:
     "result" = OOS Sharpe gain over raw H1 (filter) or OOS Sharpe (standalone); rejected if it changes sign AND the
     value version is better.  Untestable if any leg has < 20 independent events (>=5 d apart) summed over the OOS
     windows at each fold's selected parameters.
 D10 H6: depth window = folds whose NOMINAL training window (fold train_start .. test_start - 5D) contains >= 12 months
     of depth data (>= 2023-01-01); in-sample selection for H6 uses that nominal window.  Depth values = `depth` (base
     units) at +-1 % / +-2 %; day invalid if < 1,440 snapshots.  Vetoed entry = entry or flip blocked (distinct
     episodes), counted over the depth-window OOS folds at each fold's selected parameters.  Placebo uses the same
     per-fold parameters on the z-score of total depth; "captures >= 80 %" = placebo OOS Sharpe gain >= 0.8 x H6 gain.
 D11 Trials go to research/trials.duckdb (stages h1..h6, tf '1d-h' so existing 1d matrices are not disturbed).
     Amendment add-on to N: A1 +1, A2 +1, A3 +1, A4 +6 (disclosed runs) = 9.
 D12 S2 combination step runs only for components that pass individually AND the DSR gate.  S4 holdout rows are not
     evaluated (holdout never loaded; S2.6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression

from btclab.api.presets import get_preset
from btclab.data import loaders
from btclab.engine.costs import BINANCE_PERP_TAKER
from btclab.engine.metrics import daily_returns
from btclab.indicators import base as ib
from btclab.research.search.core import build_context
from btclab.research.search.trials import TrialStore
from btclab.research.stats.bootstrap import block_bootstrap
from btclab.research.stats.cscv import cscv_pbo
from btclab.research.stats.dsr import (deflated_sharpe_ratio, expected_max_sharpe, sharpe_ratio, trial_sharpes)
from btclab.research.stats.spa import spa_test
from btclab.signals.combiner import run_spec

PPY = 365.0
COST = BINANCE_PERP_TAKER.fee_rate + BINANCE_PERP_TAKER.slippage_bp / 1e4
LMAX = 2.0
GUARD = 0.10
LB = (10, 20, 40, 80, 160)
MED = dict(sw=40, tgt=0.30, ss=0.5, db=0.125)
DEPTH_START = pd.Timestamp("2023-01-01")
REPORT = ROOT / "research" / "reports" / "h1_h6_results.md"
LOG = ROOT / "research" / "h1_h6.log"
RUNS = ROOT / "research" / "runs"


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


# ---------------------------------------------------------------------------------------------- stats helpers
def sh(x) -> float:
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    s = sharpe_ratio(x, PPY)
    return float(s) if np.isfinite(s) else 0.0


def maxdd(x) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + x)
    return float((1.0 - eq / np.maximum.accumulate(eq)).max())


def tot_ret(x) -> float:
    return float(np.prod(1.0 + np.asarray(x, float)) - 1.0)


def calmar(x) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    ann = np.prod(1.0 + x) ** (PPY / len(x)) - 1.0
    dd = maxdd(x)
    if dd <= 0:
        return float("inf") if ann > 0 else 0.0
    return float(ann / dd)


# ---------------------------------------------------------------------------------------------- numba kernels
@njit
def _guard(T, thr):
    n = len(T)
    P = np.zeros(n)
    cur = 0.0
    for i in range(n):
        t = T[i]
        if t != t:
            t = 0.0
        if t == 0.0:
            cur = 0.0
        elif cur == 0.0 or (t > 0.0) != (cur > 0.0):
            cur = t
        elif abs(t - cur) > thr:
            cur = t
        P[i] = cur
    return P


@njit
def _exec(T, vl, vs, thr):
    """Guarded execution with entry vetoes. veto[i]=1: entry/flip blocked, 2: add blocked."""
    n = len(T)
    P = np.zeros(n)
    veto = np.zeros(n)
    cur = 0.0
    for i in range(n):
        t = T[i]
        if t != t:
            t = 0.0
        if t == 0.0:
            cur = 0.0
        elif cur == 0.0 or (t > 0.0) != (cur > 0.0):
            if (t > 0.0 and vl[i]) or (t < 0.0 and vs[i]):
                cur = 0.0
                veto[i] = 1.0
            else:
                cur = t
        elif abs(t) > abs(cur):
            if abs(t - cur) > thr:
                if (t > 0.0 and vl[i]) or (t < 0.0 and vs[i]):
                    veto[i] = 2.0
                else:
                    cur = t
        else:
            if abs(t - cur) > thr:
                cur = t
        P[i] = cur
    return P, veto


@njit
def _deadband(S, db):
    n = len(S)
    out = np.zeros(n)
    cur = 0.0
    for i in range(n):
        s = S[i]
        if s != s:
            s = 0.0
        if (s > 0.0) != (cur > 0.0) or (s < 0.0) != (cur < 0.0) or abs(s - cur) > db:
            cur = s
        out[i] = cur
    return out


@njit
def _mr(c, sma, sd, rsi, size, rng, blocked, z, ss):
    n = len(c)
    out = np.zeros(n)
    pos = 0
    held = 0
    for i in range(n):
        if (not rng[i]) or blocked[i]:
            pos = 0
            held = 0
            continue
        if pos != 0:
            held += 1
            if (pos > 0 and c[i] >= sma[i]) or (pos < 0 and c[i] <= sma[i]) or held >= 5:
                pos = 0
        if pos == 0:
            if c[i] < sma[i] - z * sd[i] and rsi[i] < 20.0:
                pos = 1
                held = 0
            elif c[i] > sma[i] + z * sd[i] and rsi[i] > 80.0:
                pos = -1
                held = 0
        sz = size[i]
        if sz != sz:
            sz = 0.0
        out[i] = pos * 0.5 * sz * (ss if pos < 0 else 1.0)
    return out


@njit
def _h5(fl, fs, fu, oi, c, low, size, hold, do_frag):
    """Standalone H5 engine. state 1 fragile long (short pos), 2 fragile short (long pos), 3 flush long."""
    n = len(c)
    out = np.zeros(n)
    state = 0
    count = 0
    oi0 = 0.0
    low0 = 0.0
    for t in range(n):
        if state != 0:
            ex = False
            if count >= hold:
                ex = True
            elif (state == 1 or state == 2) and oi[t] < oi0:
                ex = True
            elif state == 3 and c[t] < low0:
                ex = True
            if ex:
                state = 0
            else:
                count += 1
        if state == 0:
            if do_frag and fl[t]:
                state = 1
                oi0 = oi[t]
                count = 1
            elif do_frag and fs[t]:
                state = 2
                oi0 = oi[t]
                count = 1
            elif fu[t]:
                state = 3
                low0 = low[t]
                count = 1
        sz = size[t]
        if sz != sz:
            sz = 0.0
        if state == 1:
            out[t] = -0.5 * sz
        elif state == 2 or state == 3:
            out[t] = 0.5 * sz
    return out


@njit
def _meta(P, ev, p, pmin):
    n = len(P)
    out = np.zeros(n)
    m = 0.0
    prev = 0.0
    for i in range(n):
        b = P[i]
        if b == 0.0 or prev == 0.0 or (b > 0.0) != (prev > 0.0):
            m = 0.0
        if ev[i]:
            pi = p[i]
            if pi > pmin:
                m = min(1.0, (pi - 0.5) / 0.25)
            else:
                m = 0.0
        out[i] = b * m
        prev = b
    return out


# ---------------------------------------------------------------------------------------------- data
class D:
    pass


def load_all():
    d = D()
    d.cut = loaders.get_cutoff()
    d.cut_day = d.cut.floor("D")
    df = loaders.load_ohlcv("perp", "1d", end=d.cut_day)
    fr = loaders.load_funding("1d", end=d.cut_day)
    df = df.assign(funding=fr.reindex(df.index).fillna(0.0).to_numpy())
    d.df = df
    d.ctx = build_context("1d", df=df, cutoff=d.cut_day)
    ctx = d.ctx
    d.days = ctx.days
    assert len(d.days) == len(df)
    d.o, d.h, d.l, d.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    d.f = df["funding"].to_numpy(float)
    d.vol = df["volume"].to_numpy(float)
    d.tb = df["taker_buy_base"].to_numpy(float)
    d.bench = ctx.bench
    d.n = len(df)
    lr = np.log(pd.Series(d.c)).diff()
    d.lr = lr
    d.atr = ib.atr(d.h, d.l, d.c, 14)
    # raw funding prints (naive UTC)
    raw = loaders.load_funding(None, end=d.cut_day)
    raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    d.fprints = raw
    return d


def sigma_ann(d, w):
    return (d.lr.ewm(span=w, adjust=False, min_periods=w).std() * math.sqrt(PPY)).to_numpy()


def S_series(d):
    c = pd.Series(d.c)
    S = sum(np.sign(c - c.shift(L)) for L in LB) / len(LB)
    return S.to_numpy()


def h1_target(d, sig, Seff, tgt, ss):
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = Seff * tgt / sig
    pos = np.where(np.isfinite(pos), pos, 0.0)
    pos = np.where(pos < 0, pos * ss, pos)
    return np.clip(pos, -LMAX, LMAX)


def shift(a, k=1, fill=0.0):
    out = np.full(len(a), fill, dtype=float)
    out[k:] = a[:-k] if k > 0 else a
    return out


def net_ret(d, P):
    p1, p2 = shift(P, 1), shift(P, 2)
    gap = np.where(np.isfinite(d.o / shift(d.c, 1, np.nan) - 1.0), d.o / shift(d.c, 1, np.nan) - 1.0, 0.0)
    intra = d.c / d.o - 1.0
    return p2 * gap + p1 * intra - np.abs(p1 - p2) * COST - p1 * d.f


def run_ids(P):
    s = np.sign(P)
    prev = np.r_[0.0, s[:-1]]
    new = (s != 0) & (s != prev)
    rid = (np.cumsum(new) * (s != 0)).astype(int)
    return rid, np.flatnonzero(new)


def trade_table(d, P):
    """Per round trip: entry index, net P&L (equity fraction), return on mean notional."""
    rid, entries = run_ids(P)
    nrun = len(entries)
    if nrun == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    p1, p2 = shift(P, 1), shift(P, 2)
    rb, ra = shift(rid, 1).astype(int), shift(rid, 2).astype(int)
    gap = np.where(np.isfinite(d.o / shift(d.c, 1, np.nan) - 1.0), d.o / shift(d.c, 1, np.nan) - 1.0, 0.0)
    intra = d.c / d.o - 1.0
    ml = nrun + 1
    net = np.bincount(ra, weights=p2 * gap, minlength=ml) + np.bincount(rb, weights=p1 * intra - p1 * d.f, minlength=ml)
    same = ra == rb
    net -= np.bincount(rb, weights=np.where(same, np.abs(p1 - p2) * COST, np.abs(p1) * COST), minlength=ml)
    net -= np.bincount(ra, weights=np.where(same, 0.0, np.abs(p2) * COST), minlength=ml)
    notional = np.bincount(rid, weights=np.abs(P), minlength=ml)[1:] / np.maximum(np.bincount(rid, minlength=ml)[1:], 1)
    pnl = net[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(notional > 0, pnl / notional, 0.0)
    return entries, pnl, pct


def legs_trades(d, legs):
    parts = [trade_table(d, P) for P in legs]
    e = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, int)
    pnl = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
    pct = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0)
    return e, pnl, pct


def n_fills(P):
    return int((np.abs(np.diff(np.r_[0.0, P])) > 1e-12).sum())


def indep_events(flag, mask=None, gap=5):
    """Greedy count of events >= gap days apart among True days (optionally restricted to mask days)."""
    idx = np.flatnonzero(flag & (mask if mask is not None else True))
    cnt, last = 0, -10**9
    for i in idx:
        if i - last >= gap:
            cnt += 1
            last = i
    return cnt


# ---------------------------------------------------------------------------------------------- generic WF
def wf_eval(R, tr_masks, te_masks, IS=None):
    K, N = len(te_masks), R.shape[1]
    if IS is None:
        IS = np.array([[sh(R[m, j]) for j in range(N)] for m in tr_masks])
    OOS = np.array([[sh(R[m, j]) for j in range(N)] for m in te_masks])
    sel = IS.argmax(1)
    cand = np.full(R.shape[0], np.nan)
    for k, m in enumerate(te_masks):
        cand[m] = R[m, sel[k]]
    return dict(sel=sel, IS=IS, OOS=OOS, cand=cand, oos=np.any(te_masks, axis=0))


def fold_stats(x_full, te_masks):
    out = []
    for m in te_masks:
        x = x_full[m]
        out.append(dict(sh=sh(x), ret=tot_ret(x), dd=maxdd(x), cal=calmar(x)))
    return out


VAR_ALL = [None]


def family_stats(R, W, N_total, bench, oos):
    """DSR / PBO / SPA of the stitched candidate against its family matrix (OOS days only)."""
    M = R[oos, :]
    ok = M.std(0) > 0
    M = M[:, ok]
    cand = W["cand"][oos]
    ts = trial_sharpes(M)
    ts = ts[np.isfinite(ts)]
    var_f = float(np.var(ts, ddof=1)) if len(ts) > 1 else 0.0
    out = {}
    d_tot = deflated_sharpe_ratio(cand, n_trials=N_total, var_trial_sharpes=var_f, periods_per_year=PPY)
    d_fam = deflated_sharpe_ratio(cand, n_trials=int(R.shape[1]), var_trial_sharpes=var_f, periods_per_year=PPY)
    if VAR_ALL[0] is not None:
        out["dsr_all"] = deflated_sharpe_ratio(cand, n_trials=N_total, var_trial_sharpes=VAR_ALL[0], periods_per_year=PPY).dsr
    out.update(dsr=d_tot.dsr, sr_ann=d_tot.sr_annualised, srstar_ann=d_tot.expected_max_sr * math.sqrt(PPY),
               dsr_famN=d_fam.dsr, psr=d_tot.psr, var_f=var_f, n_obs=d_tot.n_obs)
    try:
        pb = cscv_pbo(M, n_blocks=16)
        out.update(pbo=pb.pbo, pbo_n=pb.n_trials)
    except Exception as e:
        out.update(pbo=float("nan"), pbo_n=0)
    try:
        sp = spa_test(M, benchmark=bench[oos], n_boot=2000, mean_block=10.0, seed=0)
        out.update(spa=sp.spa_pvalue, rc=sp.rc_pvalue)
    except Exception as e:
        out.update(spa=float("nan"), rc=float("nan"))
    try:
        sp1 = spa_test(cand, benchmark=bench[oos], n_boot=2000, mean_block=10.0, seed=0)
        out.update(spa1=sp1.spa_pvalue)
    except Exception:
        out.update(spa1=float("nan"))
    return out


# ---------------------------------------------------------------------------------------------- H1
def h1_grid(d, cache):
    S = cache["S"]
    cols, params, legs = [], [], []
    for sw in (20, 40, 60):
        sig = cache["sig"][sw]
        for db in (0.0, 0.125, 0.25):
            Se = _deadband(np.nan_to_num(S), db)
            for tgt in (0.2, 0.3, 0.4):
                for ss in (0.0, 0.5, 1.0):
                    P = _guard(h1_target(d, sig, Se, tgt, ss), GUARD)
                    cols.append(net_ret(d, P))
                    params.append(dict(sw=sw, tgt=tgt, ss=ss, db=db))
                    legs.append([P])
    return dict(R=np.column_stack(cols), params=params, legs=legs)


def h1_median_P(d, cache, ss=MED["ss"]):
    Se = _deadband(np.nan_to_num(cache["S"]), MED["db"])
    return h1_target(d, cache["sig"][MED["sw"]], Se, MED["tgt"], ss), Se


# ---------------------------------------------------------------------------------------------- H2
def er_series(c, n):
    c = pd.Series(c)
    num = (c - c.shift(n)).abs()
    den = c.diff().abs().rolling(n).sum()
    return (num / den).to_numpy()


def h2_grid(d, cache):
    c = d.c
    sma20 = pd.Series(c).rolling(20).mean().to_numpy()
    sd20 = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    rsi3 = ib.rsi(c, 3)
    sig = cache["sig"][MED["sw"]]
    size = MED["tgt"] / sig
    Se = _deadband(np.nan_to_num(cache["S"]), MED["db"])
    cols, params, legs, mrlegs = [], [], [], []
    for n in (10, 25, 40):
        er = er_series(c, n)
        valid = np.isfinite(er)
        for th in (0.2, 0.3, 0.4):
            trend = valid & (er > th)
            rng = valid & ~(er > th)
            flip = np.r_[False, trend[1:] != trend[:-1]]
            blocked = flip | np.r_[False, flip[:-1]]
            for z in (1.5, 2.0, 2.5):
                for ss in (0.0, 0.5, 1.0):
                    tt = h1_target(d, sig, Se, MED["tgt"], ss) * (trend & ~blocked)
                    Pt = _guard(tt, GUARD)
                    mr = _mr(c, sma20, sd20, rsi3, np.nan_to_num(size), rng, blocked, z, ss)
                    Pm = _guard(mr, GUARD)
                    cols.append(net_ret(d, Pt) + net_ret(d, Pm))
                    params.append(dict(n=n, theta=th, z=z, ss=ss))
                    legs.append([Pt, Pm])
    return dict(R=np.column_stack(cols), params=params, legs=legs)


# ---------------------------------------------------------------------------------------------- H3
def funding_z_daily(d, w):
    f = d.fprints
    z = (f - f.rolling(3 * w, min_periods=int(math.ceil(0.8 * 3 * w))).mean()) / f.rolling(
        3 * w, min_periods=int(math.ceil(0.8 * 3 * w))).std()
    zd = z.resample("1D").last()          # last print of the day (16:00) - the 00:00 print of day+1 is not known yet
    return zd.reindex(d.days).to_numpy()


def h3_grid(d, cache):
    base, _ = h1_median_P(d, cache)
    sig = cache["sig"][MED["sw"]]
    vs = 0.5 * MED["tgt"] / sig
    minlow5 = pd.Series(d.l).rolling(5).min().to_numpy()
    cols, params, legs = [], [], []
    for w in (30, 60, 90):
        zf = funding_z_daily(d, w)
        cache.setdefault("zf", {})[w] = zf
        for tau in (1.5, 2.0, 2.5, 3.0):
            trig = np.nan_to_num(zf, nan=0.0) < -tau
            trig &= d.c > minlow5
            for h in (0.0, 0.5):
                for hold in (1, 3, 5, 7):
                    t = base.copy()
                    t = np.where((zf > tau) & (base > 0), t * h, t)
                    t = np.where((zf < -tau) & (base < 0), t * h, t)
                    act = pd.Series(trig.astype(float)).rolling(hold, min_periods=1).max().to_numpy() > 0
                    t = np.clip(t + np.where(act, np.nan_to_num(vs), 0.0), -LMAX, LMAX)
                    P = _guard(t, GUARD)
                    cols.append(net_ret(d, P))
                    params.append(dict(w=w, tau=tau, h=h, hold=hold))
                    legs.append([P])
    return dict(R=np.column_stack(cols), params=params, legs=legs)


# ---------------------------------------------------------------------------------------------- H4
H4_FEATS = ["vol_z20", "er20", "fund_z30", "r1", "r7", "r30", "donch_atr", "taker7"]


def h4_features(d, cache, P1):
    c = pd.Series(d.c)
    lr = d.lr
    v20 = lr.rolling(20).std()
    vz = ((v20 - v20.rolling(180, min_periods=120).mean()) / v20.rolling(180, min_periods=120).std()).to_numpy()
    er20 = er_series(d.c, 20)
    fz = funding_z_daily(d, 30)
    r1 = c.pct_change(1).to_numpy() if False else np.log(c / c.shift(1)).to_numpy()
    r7 = np.log(c / c.shift(7)).to_numpy()
    r30 = np.log(c / c.shift(30)).to_numpy()
    hh = pd.Series(d.h).rolling(20).max().to_numpy()
    ll = pd.Series(d.l).rolling(20).min().to_numpy()
    tk = (pd.Series(d.tb).rolling(7).sum() / pd.Series(d.vol).rolling(7).sum()).to_numpy()
    sgn = np.sign(P1)
    prev = shift(P1, 1)
    same_prev = np.where(np.sign(prev) == sgn, np.abs(prev), 0.0)
    ev = (sgn != 0) & (np.abs(P1) - same_prev > 1e-9)
    dist = np.where(sgn > 0, (hh - d.c) / d.atr, (d.c - ll) / d.atr)
    X = np.column_stack([vz, er20, fz * sgn, r1 * sgn, r7 * sgn, r30 * sgn, dist, (tk - 0.5) * sgn])
    return ev, sgn, X


def h4_labels(d, ev, sgn, V):
    n = d.n
    y = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    for t in np.flatnonzero(ev):
        if t + 1 >= n or t + V >= n or not np.isfinite(d.atr[t]):
            continue
        dr, entry, a = sgn[t], d.o[t + 1], d.atr[t]
        tp, sl = entry + dr * 2 * a, entry - dr * a
        out = None
        for j in range(t + 1, t + V + 1):
            hi, lo = d.h[j], d.l[j]
            hit_tp = hi >= tp if dr > 0 else lo <= tp
            hit_sl = lo <= sl if dr > 0 else hi >= sl
            if hit_sl:
                out = 0.0
                break
            if hit_tp:
                out = 1.0
                break
        if out is None:
            out = 1.0 if dr * (d.c[t + V] / entry - 1.0) > 0 else 0.0
        y[t] = out
        ret[t] = dr * (d.c[t + V] / entry - 1.0)
    return y, ret


def _fit(Xtr, ytr, C):
    ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    X, y = Xtr[ok], ytr[ok]
    if len(y) < 20 or min(y.sum(), len(y) - y.sum()) < 4:
        return None
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=C, penalty="l2", class_weight="balanced", max_iter=1000)
    m.fit((X - mu) / sd, y)
    return m, mu, sd


def _pred(model, X):
    p = np.full(len(X), 0.5)
    if model is None:
        return p
    m, mu, sd = model
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    return m.predict_proba(Z)[:, 1]


def h4_build(d, cache, ctx):
    P1, _ = h1_median_P(d, cache)
    P1 = _guard(P1, GUARD)
    ev, sgn, X = h4_features(d, cache, P1)
    days = d.days
    K = len(ctx.test_masks)
    Cs, pms, Vs = (0.01, 0.1, 1.0), (0.50, 0.55, 0.60), (5, 10, 15)
    cfgs = [(C, pm, V) for C in Cs for pm in pms for V in Vs]
    N = len(cfgs)
    R = np.zeros((d.n, N))
    IS = np.zeros((K, N))
    legs = [[np.zeros(d.n)] for _ in range(N)]     # filled per fold below (stitched OOS position)
    info = {}                                       # (k, j) -> dict(prec, coef_share, n_taken)
    labels = {V: h4_labels(d, ev, sgn, V) for V in Vs}
    ev_idx = np.flatnonzero(ev)
    for k, f in enumerate(ctx.folds):
        te = ctx.test_masks[k]
        tr = ctx.train_masks[k]
        ts_ = pd.Timestamp(f.test_start).tz_localize(None) if pd.Timestamp(f.test_start).tzinfo else pd.Timestamp(f.test_start)
        trs = pd.Timestamp(f.train_start).tz_localize(None) if pd.Timestamp(f.train_start).tzinfo else pd.Timestamp(f.train_start)
        for V in Vs:
            y, ret = labels[V]
            cut_t = ts_ - pd.Timedelta(days=5 + max(10, V))
            tr_ev = np.array([t for t in ev_idx if trs <= days[t] <= cut_t and np.isfinite(y[t])], int)
            for C in Cs:
                # out-of-fold probabilities (3 contiguous blocks, 15-day purge)
                p_all = np.full(d.n, 0.5)
                if len(tr_ev) >= 30:
                    parts = np.array_split(tr_ev, 3)
                    for b in range(3):
                        blk = parts[b]
                        lo, hi = blk[0] - 15, blk[-1] + 15
                        tr_in = np.array([t for t in tr_ev if t < lo or t > hi], int)
                        mdl = _fit(X[tr_in], y[tr_in], C) if len(tr_in) else None
                        p_all[blk] = _pred(mdl, X[blk])
                model = _fit(X[tr_ev], y[tr_ev], C) if len(tr_ev) else None
                after = ev_idx[ev_idx > (tr_ev.max() if len(tr_ev) else -1)]
                p_all[after] = _pred(model, X[after])
                coef = np.abs(model[0].coef_[0]) if model is not None else np.full(len(H4_FEATS), np.nan)
                share = coef / np.nansum(coef) if model is not None else coef
                for pm in pms:
                    j = cfgs.index((C, pm, V))
                    Pm = _guard(_meta(P1, ev, p_all, pm), GUARD)
                    r = net_ret(d, Pm)
                    R[te, j] = r[te]
                    IS[k, j] = sh(r[tr])
                    legs[j][0] = np.where(te, Pm, legs[j][0])
                    taken = ev & te & (np.abs(Pm) > 0) & np.isfinite(y)
                    # an event is "taken" when the meta multiplier is > 0 on the event day
                    taken = ev & te & (np.abs(Pm) > 1e-12) & np.isfinite(y)
                    info[(k, j)] = dict(prec=float(y[taken].mean()) if taken.any() else float("nan"),
                                        n_taken=int(taken.sum()), n_events=int((ev & te & np.isfinite(y)).sum()),
                                        share=share, win=float((ret[taken] > 0).mean()) if taken.any() else float("nan"))
    params = [dict(C=C, p_min=pm, V=V) for (C, pm, V) in cfgs]
    return dict(R=R, params=params, legs=legs, IS=IS, info=info, ev=ev, P1=P1)


# ---------------------------------------------------------------------------------------------- H5
def load_metrics(d):
    m = pd.read_parquet(ROOT / "data/raw/um_futures/metrics_archive_5m.parquet",
                        columns=["sum_open_interest", "sum_open_interest_value", "sum_toptrader_long_short_ratio"])
    m.index = m.index.tz_convert("UTC").tz_localize(None)
    m = m[m.index < d.cut_day.tz_convert("UTC").tz_localize(None)]
    return m


def asof_daily(m, col, days):
    """A1.4: value observed at create_time <= daily close - 5 min, age <= 12 h, never interpolated."""
    s = m[col].dropna()
    t = (days + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)).to_numpy()
    left = pd.DataFrame({"t": t})
    right = pd.DataFrame({"obs": s.index.to_numpy(), "v": s.to_numpy()})
    mg = pd.merge_asof(left, right, left_on="t", right_on="obs", direction="backward")
    age = mg["t"] - mg["obs"]
    v = mg["v"].to_numpy(float)
    v[(age > pd.Timedelta(hours=12)).to_numpy() | mg["obs"].isna().to_numpy()] = np.nan
    return v


def rzs(x, w):
    s = pd.Series(x)
    mp = int(math.ceil(0.8 * w))
    return ((s - s.rolling(w, min_periods=mp).mean()) / s.rolling(w, min_periods=mp).std()).to_numpy()


def h5_prepare(d, cache):
    m = load_metrics(d)
    d.metrics = m
    oi = asof_daily(m, "sum_open_interest", d.days)
    oiv = asof_daily(m, "sum_open_interest_value", d.days)
    top = asof_daily(m, "sum_toptrader_long_short_ratio", d.days)
    d.oi, d.oiv, d.top = oi, oiv, top
    d.metrics_gap = ~(np.isfinite(oi) & np.isfinite(top))
    # days with at least one missing 5m bar (information)
    grid = pd.date_range(m.index.min(), m.index.max(), freq="5min")
    miss = grid.difference(m.index)
    d.metrics_missing_days = set(pd.DatetimeIndex(miss).normalize())
    d.metrics_gap_any = np.array([x in d.metrics_missing_days for x in d.days])
    cache["h5"] = {}
    for src, arr in (("oi", oi), ("oiv", oiv)):
        for k in (1, 3):
            dk = np.log(arr / shift(arr, k, np.nan))
            for w in (30, 60, 90):
                cache["h5"][(src, k, w)] = rzs(dk, w)
    for w in (30, 60, 90):
        cache["h5"][("top", w)] = rzs(top, w)
    for k in (1, 3):
        cache["h5"][("r", k)] = np.log(d.c / shift(d.c, k, np.nan))


def h5_grid(d, cache, mode, src):
    base, _ = h1_median_P(d, cache)
    sig = cache["sig"][MED["sw"]]
    size = np.nan_to_num(MED["tgt"] / sig)
    oi = d.oi if src == "oi" else d.oiv
    cols, params, legs, evs = [], [], [], []
    for k in (1, 3):
        rk = cache["h5"][("r", k)]
        for w in (30, 60, 90):
            zo, zt = cache["h5"][(src, k, w)], cache["h5"][("top", w)]
            for tau in (1.5, 2.0, 2.5, 3.0):
                fl = (rk > 0) & (zo > tau) & (zt > tau)
                fs = (rk < 0) & (zo > tau) & (zt < -tau)
                fu = (rk < 0) & (zo < -tau)
                for hold in (1, 3, 5, 7):
                    if mode == "standalone":
                        t = _h5(fl, fs, fu, np.nan_to_num(oi, nan=np.inf), d.c, d.l, size, hold, True)
                    else:
                        fz = _h5(fl, fs, fu, np.nan_to_num(oi, nan=np.inf), d.c, d.l, size, hold, False)
                        t = np.where(fl & (base > 0), base * 0.5, base)
                        t = np.where(fs & (base < 0), t * 0.5, t)
                        t = np.clip(t + fz, -LMAX, LMAX)
                    P = _guard(t, GUARD)
                    cols.append(net_ret(d, P))
                    params.append(dict(k=k, w=w, tau=tau, hold=hold))
                    legs.append([P])
                    evs.append(dict(fl=fl, fs=fs, fu=fu))
    return dict(R=np.column_stack(cols), params=params, legs=legs, evs=evs)


# ---------------------------------------------------------------------------------------------- H6 (depth)
def build_depth_daily(d):
    cache_f = RUNS / "h6_depth_daily.parquet"
    if cache_f.exists():
        return pd.read_parquet(cache_f)
    import pyarrow.dataset as pads
    ds_ = pads.dataset(str(ROOT / "data/raw/um_futures/bookdepth_archive.parquet"))
    end = d.cut_day
    months = pd.date_range("2023-01-01", end.tz_convert("UTC").tz_localize(None) + pd.offsets.MonthBegin(1), freq="MS", tz="UTC")
    rows = []
    for a, b in zip(months[:-1], months[1:]):
        b = min(b, end)
        if a >= b:
            continue
        flt = ((pads.field("timestamp") >= a) & (pads.field("timestamp") < b) &
               pads.field("percentage").isin([-2.0, -1.0, 1.0, 2.0]))
        t = ds_.to_table(columns=["timestamp", "percentage", "depth"], filter=flt).to_pandas()
        if t.empty:
            continue
        w = t.groupby(["timestamp", "percentage"])["depth"].last().unstack()
        for need in (-2.0, -1.0, 1.0, 2.0):
            if need not in w.columns:
                w[need] = np.nan
        out = pd.DataFrame({
            "I1": (w[-1.0] - w[1.0]) / (w[-1.0] + w[1.0]), "I2": (w[-2.0] - w[2.0]) / (w[-2.0] + w[2.0]),
            "T1": w[-1.0] + w[1.0], "T2": w[-2.0] + w[2.0]}).dropna(how="all")
        out.index = out.index.tz_convert("UTC").tz_localize(None)
        g = out.groupby(out.index.normalize())
        dd = g.median()
        dd["n"] = g.size()
        rows.append(dd)
        log("depth", a.date(), len(t))
    dd = pd.concat(rows)
    dd = dd[~dd.index.duplicated(keep="last")]
    dd.to_parquet(cache_f)
    return dd


def h6_prepare(d, cache):
    dd = build_depth_daily(d).reindex(d.days)
    valid = (dd["n"].fillna(0) >= 1440).to_numpy()
    for col in ("I1", "I2", "T1", "T2"):
        dd.loc[~valid, col] = np.nan
    d.depth = dd
    d.depth_gap = ~valid
    d.depth_gap[d.days < DEPTH_START] = True


def h6_grid(d, cache, placebo=False):
    Se = _deadband(np.nan_to_num(cache["S"]), MED["db"])
    base = h1_target(d, cache["sig"][MED["sw"]], Se, MED["tgt"], MED["ss"])
    cols, params, legs, vetos = [], [], [], []
    for b in (1, 2):
        series = d.depth[("T%d" if placebo else "I%d") % b].to_numpy()
        for w in (20, 40, 60):
            z = rzs(series, w)
            for g in (0.5, 1.0, 1.5):
                vl = np.nan_to_num(z, nan=0.0) < -g
                vs = np.nan_to_num(z, nan=0.0) > g
                P, veto = _exec(base, vl, vs, GUARD)
                cols.append(net_ret(d, P))
                params.append(dict(b=b, w=w, gamma=g))
                legs.append([P])
                vetos.append(veto)
    return dict(R=np.column_stack(cols), params=params, legs=legs, vetos=vetos)


# ---------------------------------------------------------------------------------------------- orchestration
def per_fold_table(d, name, W, R, legs, te_masks, fold_ids, raw, bh_f, bray_f, extra_cols=None):
    rows = []
    cand_f = fold_stats(W["cand"], te_masks)
    for i, m in enumerate(te_masks):
        j = W["sel"][i]
        e, pnl, pct = legs_trades(d, legs[j])
        ntr = int(m[e].sum())
        rows.append(dict(fold=fold_ids[i], window=f"{d.days[m][0].date()}..{d.days[m][-1].date()}", sel=j,
                         sh=cand_f[i]["sh"], raw=raw[i]["sh"], bh=bh_f[i]["sh"], bray=bray_f[i]["sh"],
                         ret=cand_f[i]["ret"], dd=cand_f[i]["dd"], cal=cand_f[i]["cal"], bh_cal=bh_f[i]["cal"],
                         trades=ntr))
    return rows, cand_f


def candidate_trades(d, W, legs, te_masks, tr_masks):
    """OOS round trips of the stitched candidate: entries, pct returns; IS trade union; per-fold OOS counts."""
    oos_pct, per_fold, is_ct = [], [], 0
    mr_pct = []
    for i, m in enumerate(te_masks):
        j = W["sel"][i]
        e, pnl, pct = legs_trades(d, legs[j])
        sel = m[e]
        oos_pct.append(pct[sel])
        per_fold.append(int(sel.sum()))
        is_ct += int(tr_masks[i][e].sum())
    return np.concatenate(oos_pct) if oos_pct else np.zeros(0), per_fold, is_ct


def funding_paid(d, P_legs_total, te_mask):
    """Annualised funding paid (fraction of equity per year) over OOS days."""
    p1 = shift(P_legs_total, 1)
    return float((p1 * d.f)[te_mask].mean() * PPY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    RUNS.mkdir(parents=True, exist_ok=True)
    log("start; loading data")
    d = load_all()
    ctx = d.ctx
    K = len(ctx.folds)
    log("cutoff", d.cut, "cut_day", d.cut_day, "bars", d.n, "last day", d.days[-1].date())
    for f in ctx.folds:
        log(repr(f))
    prereg = ROOT / "docs" / "preregistration.md"
    pre_hash = hashlib.sha256(prereg.read_bytes()).hexdigest()

    cache = {"S": S_series(d), "sig": {w: sigma_ann(d, w) for w in (20, 40, 60)}}
    tr_all, te_all = ctx.train_masks, ctx.test_masks
    oos_all = ctx.oos_mask
    bench = d.bench

    # ----- benchmarks on the full folds
    Pmed, _ = h1_median_P(d, cache)
    Pmed = _guard(Pmed, GUARD)
    raw_ret = net_ret(d, Pmed)
    bh_ret = bench
    log("brayvb_v2 default (4h)")
    df4 = loaders.load_ohlcv("perp", "4h", end=d.cut_day)
    df4 = df4.assign(funding=loaders.load_funding("4h", end=d.cut_day).reindex(df4.index).fillna(0.0).to_numpy())
    res4, _ = run_spec(get_preset("brayvb_v2"), df4)
    bray_ret = daily_returns(res4.equity, df4.index).reindex(d.days).fillna(0.0).to_numpy()
    bray_trades = res4.trades

    # ----- DB counts
    store0 = TrialStore()
    n_before = store0.count()
    n_h_old = int(store0.con.execute("SELECT count(*) FROM trials WHERE stage IN ('h1','h2','h3','h4','h5','h6')").fetchone()[0])
    n_base = n_before - n_h_old
    xs = np.array([r[0] for r in store0.con.execute(
        "SELECT oos_sharpe FROM trials WHERE stage NOT IN ('h1','h2','h3','h4','h5','h6') AND oos_sharpe IS NOT NULL").fetchall()], float)
    VAR_ALL[0] = float(np.nanvar(xs, ddof=1) / PPY)
    log("all-trials Sharpe variance (per period)", VAR_ALL[0], "n", len(xs))
    store0.close()
    log("trials table before:", n_before, "of which previous h1-h6:", n_h_old, "base", n_base)

    fams = {}

    def run_family(name, fam, tfmasks=None, **kw):
        log("family", name)
        return fam

    # ================= H1
    g1 = h1_grid(d, cache)
    fams["h1"] = dict(g=g1, tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all)
    # ================= H2
    fams["h2"] = dict(g=h2_grid(d, cache), tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all)
    log("H1,H2 done")
    # ================= H3
    fams["h3"] = dict(g=h3_grid(d, cache), tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all)
    log("H3 done")
    # ================= H4
    g4 = h4_build(d, cache, ctx)
    fams["h4"] = dict(g=g4, tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all, IS=g4["IS"])
    log("H4 done")
    # ================= H5
    h5_prepare(d, cache)
    fams["h5f"] = dict(g=h5_grid(d, cache, "filter", "oi"), tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all)
    fams["h5s"] = dict(g=h5_grid(d, cache, "standalone", "oi"), tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all)
    fams["h5fv"] = dict(g=h5_grid(d, cache, "filter", "oiv"), tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all)
    fams["h5sv"] = dict(g=h5_grid(d, cache, "standalone", "oiv"), tr=tr_all, te=te_all, ids=list(range(K)), oos=oos_all)
    log("H5 done")
    # ================= H6 depth window
    h6_prepare(d, cache)
    keep, dropped = [], []
    for k, f in enumerate(ctx.folds):
        ns = pd.Timestamp(f.train_start).tz_localize(None) if pd.Timestamp(f.train_start).tzinfo else pd.Timestamp(f.train_start)
        ts_ = pd.Timestamp(f.test_start).tz_localize(None) if pd.Timestamp(f.test_start).tzinfo else pd.Timestamp(f.test_start)
        nom_end = ts_ - pd.Timedelta(days=5)
        eff_start = max(ns, DEPTH_START)
        months = (nom_end - eff_start).days / 30.4375
        (keep if months >= 12 else dropped).append((k, round(months, 1)))
    dk = [k for k, _ in keep]
    kd = len(dk)
    thr_d = math.ceil(2 * kd / 3)
    tr6, te6 = [], []
    for k in dk:
        f = ctx.folds[k]
        ns = pd.Timestamp(f.train_start).tz_localize(None) if pd.Timestamp(f.train_start).tzinfo else pd.Timestamp(f.train_start)
        ts_ = pd.Timestamp(f.test_start).tz_localize(None) if pd.Timestamp(f.test_start).tzinfo else pd.Timestamp(f.test_start)
        tr6.append((d.days >= max(ns, DEPTH_START)) & (d.days < ts_ - pd.Timedelta(days=5)))
        te6.append(te_all[k])
    oos6 = np.any(te6, axis=0)
    fams["h6"] = dict(g=h6_grid(d, cache), tr=tr6, te=te6, ids=dk, oos=oos6)
    fams["h6p"] = dict(g=h6_grid(d, cache, placebo=True), tr=tr6, te=te6, ids=dk, oos=oos6)
    log("H6 done; depth folds", keep, "dropped", dropped)

    # ----- walk-forward selection per family
    for name, F in fams.items():
        F["W"] = wf_eval(F["g"]["R"], F["tr"], F["te"], IS=F.get("IS") if name != "h4" else F["IS"])
        F["R"] = F["g"]["R"]
        F["N"] = F["R"].shape[1]
    # note: h4 IS matrix is K x N from fold-specific simulations (already built)

    # ----- N accounting
    new_trials_total = sum(F["N"] for F in fams.values())
    a1_new = sum(fams[k]["N"] for k in ("h5f", "h5s", "h5fv", "h5sv", "h6", "h6p"))
    amend = 9
    N_with = n_base + new_trials_total + amend
    N_without = n_base + (new_trials_total - a1_new) + (amend - 1)
    log("N with A1", N_with, "without A1", N_without)

    # ----- benchmarks per family folds
    def bm(F):
        raw = fold_stats(raw_ret, F["te"])
        bh = fold_stats(bh_ret, F["te"])
        bray = fold_stats(bray_ret, F["te"])
        return raw, bh, bray

    # ----- evaluate every family
    EV = {}
    for name, F in fams.items():
        g, W = F["g"], F["W"]
        raw, bh, bray = bm(F)
        rows, cand_f = per_fold_table(d, name, W, F["R"], g["legs"], F["te"], F["ids"], raw, bh, bray)
        stats = family_stats(F["R"], W, N_with, bench, F["oos"])
        pct_oos, per_fold_tr, is_ct = candidate_trades(d, W, g["legs"], F["te"], F["tr"])
        cand_oos = W["cand"][F["oos"]]
        is_sel = float(np.mean([W["IS"][i, W["sel"][i]] for i in range(len(F["te"]))]))
        oos_sh = sh(cand_oos)
        ev = dict(rows=rows, cand_f=cand_f, raw=raw, bh=bh, bray=bray, stats=stats, pct_oos=pct_oos,
                  per_fold_tr=per_fold_tr, is_ct=is_ct, is_sh=is_sel, oos_sh=oos_sh,
                  oos_ret=tot_ret(cand_oos), oos_dd=maxdd(cand_oos),
                  med_sh=float(np.median([r["sh"] for r in rows])),
                  n_pos=int(sum(r["sh"] > 0 for r in rows)), K=len(rows))
        # funding paid
        tot_leg = sum(shift(P, 1) * d.f for P in [np.zeros(d.n)] ) if False else None
        fp = []
        for i, m in enumerate(F["te"]):
            j = W["sel"][i]
            Ptot = np.sum(g["legs"][j], axis=0)
            fp.append(funding_paid(d, Ptot, m))
        ev["fund_paid_yr"] = float(np.mean(fp))
        ev["exp_bp"] = float(np.mean(pct_oos) * 1e4) if len(pct_oos) else float("nan")
        ev["n_rt"] = int(len(pct_oos))
        ev["a2"] = (is_ct >= 40) and (min(per_fold_tr) >= 8)
        EV[name] = ev
        log(name, "oos_sh", round(oos_sh, 3), "med", round(ev["med_sh"], 3), "npos", ev["n_pos"], "dsr", round(stats["dsr"], 4),
            "pbo", round(stats["pbo"], 3), "spa", round(stats["spa"], 3), "rt", ev["n_rt"])

    # ----- hypothesis-specific falsification
    V = {}
    # H1
    e1 = EV["h1"]
    calwin = sum(1 for r in e1["rows"] if r["cal"] > r["bh_cal"])
    fails = []
    if e1["n_pos"] < 4:
        fails.append(f"OOS Sharpe > 0 in only {e1['n_pos']}/6 folds")
    if calwin < 4:
        fails.append(f"beats B&H Calmar in only {calwin}/6 folds")
    V["H1"] = dict(status="REJECTED" if fails else "PASS", why="; ".join(fails), tests=[
        ("OOS Sharpe>0 in >=4/6 folds", f"{e1['n_pos']}/6", e1["n_pos"] >= 4),
        ("OOS Calmar > B&H in >=4/6 folds", f"{calwin}/6", calwin >= 4)], ev=e1, fam="h1")

    # H2
    e2, F2 = EV["h2"], fams["h2"]
    adj_wins, adj_rows = 0, []
    for i, m in enumerate(F2["te"]):
        col = F2["W"]["OOS"][i] / math.sqrt(PPY)          # per-period fold Sharpe of every config
        vf = float(np.nanvar(col, ddof=1))
        sstar = expected_max_sharpe(F2["N"], vf) * math.sqrt(PPY)
        adj = e2["rows"][i]["sh"] - sstar
        adj_rows.append((adj, sstar, e2["raw"][i]["sh"]))
        adj_wins += adj > e2["raw"][i]["sh"]
    # MR leg expectancy at the selected configs
    mr_pct = []
    for i, m in enumerate(F2["te"]):
        j = F2["W"]["sel"][i]
        e, pnl, pct = trade_table(d, F2["g"]["legs"][j][1])
        mr_pct.append(pct[m[e]])
    mr_pct = np.concatenate(mr_pct)
    mr_exp = float(mr_pct.mean() * 1e4) if len(mr_pct) else float("nan")
    fails = []
    if adj_wins < 4:
        fails.append(f"DSR-adjusted OOS Sharpe above H1 alone in only {adj_wins}/6 folds")
    if not (len(mr_pct) and mr_exp > 0):
        fails.append(f"mean-reversion leg OOS net expectancy {mr_exp:.1f} bp over {len(mr_pct)} trades (<= 0)")
    V["H2"] = dict(status="REJECTED" if fails else "PASS", why="; ".join(fails), tests=[
        ("DSR-adj OOS Sharpe > H1 alone in >=4/6 folds", f"{adj_wins}/6", adj_wins >= 4),
        ("MR-leg OOS net expectancy > 0", f"{mr_exp:.1f} bp ({len(mr_pct)} trades)", bool(len(mr_pct) and mr_exp > 0))],
        ev=e2, fam="h2", extra=dict(adj_rows=adj_rows, mr_exp=mr_exp, mr_n=len(mr_pct)))

    # H3
    e3, F3 = EV["h3"], fams["h3"]
    ext = 0
    ext_tab = {}
    for i, m in enumerate(F3["te"]):
        p = F3["g"]["params"][F3["W"]["sel"][i]]
        zf = cache["zf"][p["w"]]
        ext += indep_events(np.nan_to_num(np.abs(zf)) > p["tau"], m)
    for w in (30, 60, 90):
        for tau in (1.5, 2.0, 2.5, 3.0):
            ext_tab[(w, tau)] = indep_events(np.nan_to_num(np.abs(cache["zf"][w])) > tau, oos_all)
    beat3 = sum(1 for r in e3["rows"] if r["sh"] > r["raw"])
    if ext < 20:
        V["H3"] = dict(status="UNTESTABLE", why=f"{ext} independent extremes (<20) in the WF OOS windows at the selected (w,tau)",
                       tests=[("independent extremes >= 20", str(ext), False), ("beats raw H1 OOS Sharpe in >=4/6 folds", f"{beat3}/6", beat3 >= 4)],
                       ev=e3, fam="h3", extra=dict(ext=ext, ext_tab=ext_tab))
    else:
        V["H3"] = dict(status="PASS" if beat3 >= 4 else "REJECTED",
                       why="" if beat3 >= 4 else f"beats raw H1 OOS Sharpe in only {beat3}/6 folds",
                       tests=[("independent extremes >= 20", str(ext), True), ("beats raw H1 OOS Sharpe in >=4/6 folds", f"{beat3}/6", beat3 >= 4)],
                       ev=e3, fam="h3", extra=dict(ext=ext, ext_tab=ext_tab))

    # H4
    e4, F4 = EV["h4"], fams["h4"]
    beat4 = sum(1 for r in e4["rows"] if r["sh"] > r["raw"])
    taken = sum(F4["g"]["info"][(i, F4["W"]["sel"][i])]["n_taken"] for i in range(K))
    prec_num = sum(F4["g"]["info"][(i, F4["W"]["sel"][i])]["prec"] * F4["g"]["info"][(i, F4["W"]["sel"][i])]["n_taken"]
                   for i in range(K) if F4["g"]["info"][(i, F4["W"]["sel"][i])]["n_taken"] > 0)
    prec = prec_num / taken if taken else float("nan")
    base_rate = np.nan
    shares = [float(np.nanmax(F4["g"]["info"][(i, F4["W"]["sel"][i])]["share"])) if np.isfinite(F4["g"]["info"][(i, F4["W"]["sel"][i])]["share"]).any() else float("nan") for i in range(K)]
    mean_share = float(np.nanmean(shares)) if np.isfinite(shares).any() else float("nan")
    fails = []
    if beat4 < 4:
        fails.append(f"beats unfiltered H1 OOS Sharpe in only {beat4}/6 folds")
    if not (np.isfinite(prec) and prec > 0.52):
        fails.append(f"OOS precision of taken trades {prec:.1%} (<= 52%) on {taken} taken events")
    if np.isfinite(mean_share) and mean_share > 0.6:
        fails.append(f"largest feature carries {mean_share:.0%} of |coef| mass (> 60%, leak)")
    # event base rate (all events OOS, V=10)
    ev_all = F4["g"]["ev"]
    yb, _ = h4_labels(d, ev_all, np.sign(F4["g"]["P1"]), 10)
    base_rate = float(np.nanmean(yb[ev_all & oos_all & np.isfinite(yb)]))
    V["H4"] = dict(status="REJECTED" if fails else "PASS", why="; ".join(fails), tests=[
        ("OOS Sharpe > unfiltered H1 in >=4/6 folds", f"{beat4}/6", beat4 >= 4),
        ("OOS precision on taken trades > 52%", f"{prec:.1%} ({taken} taken; base rate of all events {base_rate:.1%})", bool(np.isfinite(prec) and prec > 0.52)),
        ("no feature > 60% of |coef| mass (mean of per-fold max)", f"{mean_share:.0%}; per fold {', '.join(f'{s:.0%}' for s in shares)}", not (np.isfinite(mean_share) and mean_share > 0.6))],
        ev=e4, fam="h4", extra=dict(taken=taken, prec=prec, shares=shares, base_rate=base_rate))

    # H5 (two modes)
    def h5_events_count(F, key):
        cnt = {"fragile longs": 0, "fragile shorts": 0, "flush": 0}
        for i, m in enumerate(F["te"]):
            j = F["W"]["sel"][i]
            evd = F["g"]["evs"][j]
            cnt["fragile longs"] += indep_events(evd["fl"], m)
            cnt["fragile shorts"] += indep_events(evd["fs"], m)
            cnt["flush"] += indep_events(evd["fu"], m)
        return cnt

    for key, vkey, mode in (("h5f", "h5fv", "filter"), ("h5s", "h5sv", "standalone")):
        e, F = EV[key], fams[key]
        ev_v = EV[vkey]
        cnt = h5_events_count(F, key)
        unt = [k for k, v in cnt.items() if v < 20]
        if mode == "filter":
            gain = e["oos_sh"] - sh(raw_ret[F["oos"]])
            gain_v = ev_v["oos_sh"] - sh(raw_ret[F["oos"]])
            beat = sum(1 for r in e["rows"] if r["sh"] > r["raw"])
            crit = ("beats raw H1 OOS Sharpe in >=4/6 folds", f"{beat}/6", beat >= 4)
            res_a, res_b = gain, gain_v
            failmsg = f"beats raw H1 OOS Sharpe in only {beat}/6 folds"
            okc = beat >= 4
        else:
            beat = e["n_pos"]
            exp_bp = e["exp_bp"]
            crit = ("OOS Sharpe > 0 in >=4/6 folds", f"{e['n_pos']}/6", e["n_pos"] >= 4)
            res_a, res_b = e["oos_sh"], ev_v["oos_sh"]
            okc = e["n_pos"] >= 4 and np.isfinite(exp_bp) and exp_bp >= 50
            failmsg = "; ".join(x for x in [
                f"OOS Sharpe > 0 in only {e['n_pos']}/6 folds" if e["n_pos"] < 4 else "",
                f"net expectancy {exp_bp:.1f} bp per round trip < 50 bp ({e['n_rt']} trades)" if not (np.isfinite(exp_bp) and exp_bp >= 50) else ""] if x)
        reversed_better = (np.sign(res_a) != np.sign(res_b)) and (res_b > res_a)
        tests = [crit]
        if mode == "standalone":
            tests.append(("net expectancy per round trip >= 50 bp", f"{e['exp_bp']:.1f} bp ({e['n_rt']} trades)", bool(np.isfinite(e['exp_bp']) and e['exp_bp'] >= 50)))
        tests.append(("value-variant check (sign reversal AND value better => reject)",
                      f"result OI {res_a:+.2f} vs OI-value {res_b:+.2f}", not reversed_better))
        tests.append(("independent events per leg >= 20 (OOS windows, selected params)",
                      ", ".join(f"{k}: {v}" for k, v in cnt.items()), not unt))
        if unt:
            status, why = "UNTESTABLE", "legs with < 20 independent events: " + ", ".join(f"{k} ({cnt[k]})" for k in unt)
            if not okc:
                why += f" (the individual criteria would also fail: {failmsg})"
        elif not okc:
            status, why = "REJECTED", failmsg
        elif reversed_better:
            status, why = "REJECTED", "result reverses sign with open-interest value and the value version is better"
        else:
            status, why = "PASS", ""
        V["H5-" + mode] = dict(status=status, why=why, tests=tests, ev=e, fam=key,
                               extra=dict(cnt=cnt, res=(res_a, res_b), ev_v=ev_v))

    # H6
    e6, F6, e6p, F6p = EV["h6"], fams["h6"], EV["h6p"], fams["h6p"]
    beat6 = sum(1 for r in e6["rows"] if r["sh"] > r["raw"])
    raw6 = sh(raw_ret[F6["oos"]])
    gain6 = e6["oos_sh"] - raw6
    gainp = e6p["oos_sh"] - raw6
    nveto = 0
    for i, m in enumerate(F6["te"]):
        vv = F6["g"]["vetos"][F6["W"]["sel"][i]] == 1.0
        ep = vv & ~np.r_[False, vv[:-1]]
        nveto += int((ep & m).sum())
    fails = []
    if nveto < 20:
        status = "UNTESTABLE"
        why = f"{nveto} entries vetoed across the depth-window OOS folds (< 20)"
    else:
        if beat6 < thr_d:
            fails.append(f"beats raw H1 OOS Sharpe in only {beat6}/{kd} folds (needs {thr_d})")
        if gain6 > 0 and gainp >= 0.8 * gain6:
            fails.append(f"placebo (total depth) captures {gainp / gain6:.0%} of H6's Sharpe gain (>= 80%)")
        status, why = ("REJECTED", "; ".join(fails)) if fails else ("PASS", "")
    se_factor = math.sqrt(36.0 / (kd * 6.0))
    V["H6"] = dict(status=status, why=why, tests=[
        (f"beats raw H1 (same depth window) OOS Sharpe in >= {thr_d}/{kd} folds", f"{beat6}/{kd}", beat6 >= thr_d),
        ("placebo captures < 80% of H6 Sharpe gain", f"H6 gain {gain6:+.2f}, placebo gain {gainp:+.2f}" + (f" ({gainp / gain6:.0%})" if gain6 > 0 else " (H6 gain <= 0)"), not (gain6 > 0 and gainp >= 0.8 * gain6)),
        ("vetoed entries >= 20 in depth OOS", str(nveto), nveto >= 20)], ev=e6, fam="h6",
        extra=dict(kd=kd, thr=thr_d, se=se_factor, months=kd * 6, keep=keep, dropped=dropped, nveto=nveto, gain6=gain6, gainp=gainp, e6p=e6p))

    # ----- gates (S2.1 eligibility, A2, S3)
    for name, v in V.items():
        e = v["ev"]
        st = e["stats"]
        v["dsr_ok"] = bool(np.isfinite(st["dsr"]) and st["dsr"] > 0.95)
        wf_thr = 4 if v["fam"] != "h6" else thr_d
        v["wf_ok"] = e["n_pos"] >= wf_thr
        v["elig"] = v["status"] == "PASS" and v["dsr_ok"] and v["wf_ok"]
        v["a2"] = e["a2"]
        v["s3"] = bool(np.isfinite(e["exp_bp"]) and e["exp_bp"] >= 50 and e["fund_paid_yr"] <= 0.15)

    # ================= report
    write_report(d, ctx, fams, EV, V, dict(pre_hash=pre_hash, n_before=n_before, n_base=n_base, N_with=N_with, N_without=N_without,
                                         new=new_trials_total, a1_new=a1_new, keep=keep, dropped=dropped, thr_d=thr_d, kd=kd,
                                         bray_trades=bray_trades), raw_ret, bh_ret, bray_ret, bench, t0)
    # ================= DB
    if not a.no_db:
        write_trials(d, ctx, fams, EV, N_with)
    log("done", round(time.time() - t0), "s")


# ---------------------------------------------------------------------------------------------- DB
def write_trials(d, ctx, fams, EV, N_with):
    store = TrialStore()
    store.con.execute("DELETE FROM trials WHERE stage IN ('h1','h2','h3','h4','h5','h6')")
    stage_of = {"h1": "h1", "h2": "h2", "h3": "h3", "h4": "h4", "h5f": "h5", "h5s": "h5", "h5fv": "h5", "h5sv": "h5", "h6": "h6", "h6p": "h6"}
    recs = []
    for name, F in fams.items():
        g, R = F["g"], F["R"]
        for j, p in enumerate(g["params"]):
            spec = dict(hyp=name, params=p)
            sj = json.dumps(spec, sort_keys=True)
            folds, isl = [], []
            for i, m in enumerate(F["te"]):
                e, _, _ = legs_trades(d, g["legs"][j])
                folds.append(dict(fold=F["ids"][i], oos_sharpe=sh(R[m, j]), oos_ret=tot_ret(R[m, j]),
                                  oos_trades=int(m[e].sum()), is_sharpe=float(F["W"]["IS"][i, j])))
            oos = R[F["oos"], j]
            e, _, _ = legs_trades(d, g["legs"][j])
            allin = np.zeros(d.n, bool)
            for m in F["tr"]:
                allin |= m
            pos = oos[oos > 0].sum()
            neg = -oos[oos < 0].sum()
            recs.append(dict(stage=stage_of[name], tf="1d-h" if name not in ("h6", "h6p") else "1d-depth",
                             spec_json=sj, spec_hash=hashlib.sha1((name + sj).encode()).hexdigest(), family=name, tag=name,
                             fold_json=json.dumps(folds), median_oos_sharpe=float(np.median([f["oos_sharpe"] for f in folds])),
                             mean_is_sharpe=float(F["W"]["IS"][:, j].mean()), n_pos_folds=int(sum(f["oos_sharpe"] > 0 for f in folds)),
                             is_trades=int(allin[e].sum()), min_oos_trades=int(min(f["oos_trades"] for f in folds)),
                             oos_sharpe=sh(oos), max_dd=maxdd(oos), profit_factor=float(min(pos / neg, 99.0)) if neg > 0 else 0.0,
                             oos_returns=oos.astype(np.float32).tobytes()))
    store.insert_many(recs)
    n = store.count()
    store.close()
    log("DB rows now", n, "(inserted", len(recs), ")")


# ---------------------------------------------------------------------------------------------- report
def f2(x, p=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def pct(x, p=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{p}f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def flag_counts(d, ctx, key):
    arr = {"metrics": d.metrics_gap, "depth": d.depth_gap}[key]
    rows = []
    for k, (tr, te) in enumerate(zip(ctx.train_masks, ctx.test_masks)):
        n = int(te.sum())
        g = int(arr[te].sum())
        rows.append((k, n, g, f"{g / n:.1%}", "DEGRADED" if g / n > 0.05 else "ok"))
    return rows


def write_report(d, ctx, fams, EV, V, meta, raw_ret, bh_ret, bray_ret, bench, t0):
    L = []
    P = L.append
    doc = (__doc__ or "")
    decisions = doc[doc.index("OPERATIONAL DECISIONS"):]
    names = ["H1", "H2", "H3", "H4", "H5-filter", "H5-standalone", "H6"]
    P("# H1-H6 pre-registered hypothesis evaluation")
    P("")
    P(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `research/h1_h6.py`; preregistration sha256 `{meta['pre_hash']}`; holdout never loaded "
      f"(data cut at the cutoff day {d.cut_day.date()}, daily bars through {d.days[-1].date()}).")
    P("")
    P("## Verdicts")
    P("")
    for n in names:
        v = V[n]
        e = v["ev"]
        gates = (f"DSR p={f2(e['stats']['dsr'], 3)} ({'>' if v['dsr_ok'] else '<='} 0.95), WF {e['n_pos']}/{e['K']} folds>0, "
                 f"A2 trade minimums {'ok' if v['a2'] else 'FAIL'} (IS {e['is_ct']}, per-fold OOS min {min(e['per_fold_tr'])}), "
                 f"S3 (>=50 bp & funding<=15%/yr) {'ok' if v['s3'] else 'FAIL'}")
        line = f"- **{n}: {v['status']}**" + (f" - {v['why']}" if v["why"] else " - all registered falsification criteria survived")
        line += f". S2.1 eligible for combination: **{'YES' if v['elig'] else 'NO'}**. [{gates}]"
        if n == "H6":
            x = v["extra"]
            line += f" Depth window: {x['kd']} folds {[k for k, _ in x['keep']]}, {x['months']} OOS months, SE of Sharpe x{x['se']:.2f} vs full window; applied threshold {x['thr']}/{x['kd']}."
        P(line)
    P("")
    n_pass = sum(1 for n in names if V[n]["status"] == "PASS")
    n_el = sum(1 for n in names if V[n]["elig"])
    P(f"**Bottom line:** {n_pass} of 7 hypothesis variants survive their own falsification criteria; {n_el} are eligible for the S2 combination step "
      f"(falsification pass + WF + family DSR p > 0.95 at N = {meta['N_with']:,}).")
    P("")
    P("## Trial accounting (S5.1, A1.3)")
    P("")
    P(f"- Trials table before this run: {meta['n_before']:,} rows (excluding earlier h1-h6 rows: {meta['n_base']:,}). New rows from this run: {meta['new']} "
      f"(of which H5/H6 additions under A1: {meta['a1_new']}). Amendment add-on: 9 (A1 +1, A2 +1, A3 +1, A4 +6).")
    P(f"- **DSR N with A1 additions: {meta['N_with']:,}; without A1 additions: {meta['N_without']:,}** (inflation {meta['N_with'] - meta['N_without']}).")
    P("")
    P("## Decisions taken before the run (ambiguities in the pre-registration)")
    P("")
    P("```")
    P(decisions.strip())
    P("```")
    P("")
    P("## Benchmarks (same folds)")
    P("")
    rows = []
    for k in range(len(ctx.folds)):
        m = ctx.test_masks[k]
        rows.append((k, f"{d.days[m][0].date()}..{d.days[m][-1].date()}", f2(sh(raw_ret[m])), f2(sh(bh_ret[m])), f2(sh(bray_ret[m])),
                     pct(tot_ret(raw_ret[m])), pct(tot_ret(bh_ret[m])), pct(tot_ret(bray_ret[m]))))
    P(table(["fold", "OOS window", "raw H1 Sharpe", "B&H Sharpe", "brayvb_v2 Sharpe", "raw H1 ret", "B&H ret", "brayvb ret"], rows))
    P("")
    P(f"brayvb_v2 default (4h preset, BINANCE_PERP_TAKER + funding): {len(meta['bray_trades'])} trades over the whole history. Raw H1 = H1 at median parameters {MED}.")
    P("")
    if True:
        P("## A1.4 gap flags (days flagged per WF test fold)")
        P("")
        P("Metrics series (H5):")
        P("")
        P(table(["fold", "OOS days", "metrics_gap days", "share", "status"], flag_counts(d, ctx, "metrics")))
        P("")
        P("Depth series (H6; days before 2023-01-01 count as gap, folds outside the depth window are not used):")
        P("")
        P(table(["fold", "OOS days", "depth_gap days", "share", "status"], flag_counts(d, ctx, "depth")))
        P("")

    descr = {
        "H1": "Volatility-managed multi-horizon TSMOM", "H2": "Regime switch (ER): trend / mean reversion",
        "H3": "Funding-rate crowding filter on H1", "H4": "Meta-labeling on H1 (L2 logit)",
        "H5-filter": "OI/price divergence filter on H1 (+ flush leg added)", "H5-standalone": "OI/price divergence standalone engine",
        "H6": "Book-depth imbalance entry filter on H1 (depth window)"}
    for n in names:
        v = V[n]
        e = v["ev"]
        F = fams[v["fam"]]
        st = e["stats"]
        P(f"## {n} - {descr[n]}: {v['status']}")
        P("")
        if v["why"]:
            P(f"Failing criterion: {v['why']}")
            P("")
        rows = []
        for r in e["rows"]:
            pr = F["g"]["params"][r["sel"]]
            rows.append((r["fold"], r["window"], ", ".join(f"{k}={x}" for k, x in pr.items()), f2(r["sh"]), f2(r["raw"]), f2(r["bh"]), f2(r["bray"]),
                         pct(r["ret"]), pct(r["dd"]), f2(r["cal"]), f2(r["bh_cal"]), r["trades"]))
        P(table(["fold", "OOS window", "selected (IS-best) params", "Sharpe", "raw H1", "B&H", "brayvb_v2", "return", "max DD", "Calmar", "B&H Calmar", "round trips"], rows))
        P("")
        bh_o = bench[F["oos"]]
        P(f"- Median fold Sharpe {f2(e['med_sh'])} (raw H1 {f2(float(np.median([r['raw'] for r in e['rows']])))}, B&H {f2(float(np.median([r['bh'] for r in e['rows']])))}, "
          f"brayvb_v2 {f2(float(np.median([r['bray'] for r in e['rows']])))}); pooled OOS Sharpe {f2(e['oos_sh'])} (IS mean {f2(e['is_sh'])}), "
          f"OOS return {pct(e['oos_ret'])}, max DD {pct(e['oos_dd'])}; B&H pooled Sharpe {f2(sh(bh_o))}, max DD {pct(maxdd(bh_o))}; raw H1 pooled {f2(sh(raw_ret[F['oos']]))}.")
        P(f"- DSR p (N={meta['N_with']:,}, family var): **{f2(st['dsr'], 4)}** (SR* = {f2(st['srstar_ann'])} annualised vs candidate {f2(st['sr_ann'])}); "
          f"with the all-trials Sharpe variance (as stages.family_stats over the broad pool): {f2(st.get('dsr_all'), 4)}; with N = family size only ({F['N']}): {f2(st['dsr_famN'], 4)}; PSR vs 0: {f2(st['psr'], 3)}.")
        P(f"- PBO (CSCV, 16 blocks, {st['pbo_n']} configs): **{f2(st['pbo'], 3)}** (gate < 0.15); SPA p vs B&H on the family: {f2(st['spa'], 3)} "
          f"(RC {f2(st['rc'], 3)}); stitched candidate alone vs B&H: {f2(st['spa1'], 3)}.")
        P(f"- Round trips: {e['n_rt']} OOS, mean net {f2(e['exp_bp'], 1)} bp each (cost per round trip 14 bp); A2: IS {e['is_ct']} (min 40), per-fold OOS {e['per_fold_tr']} (min 8); "
          f"funding paid {pct(e['fund_paid_yr'])}/yr on the OOS windows.")
        if n == "H2":
            x = v["extra"]
            P(f"- DSR-adjusted fold Sharpe (fold Sharpe - family expected max): {', '.join(f2(a[0]) for a in x['adj_rows'])}; haircuts {', '.join(f2(a[1]) for a in x['adj_rows'])}; "
              f"raw H1 {', '.join(f2(a[2]) for a in x['adj_rows'])}.")
            P(f"- MR-leg OOS expectancy {f2(x['mr_exp'], 1)} bp over {x['mr_n']} trades.")
        if n == "H3":
            x = v["extra"]
            P(f"- Independent extremes summed over OOS windows at selected params: {x['ext']}. In the OOS windows at every grid point: "
              + "; ".join(f"w{w}/tau{t}: {c}" for (w, t), c in x["ext_tab"].items()))
        if n == "H4":
            x = v["extra"]
            P(f"- Taken OOS events {x['taken']}, precision {pct(x['prec'])} vs base rate of all OOS events {pct(x['base_rate'])}; max |coef| share per fold {x['shares']}.")
        if n.startswith("H5"):
            x = v["extra"]
            P(f"- Independent events per leg (OOS windows, selected params): {x['cnt']}. Value-variant OOS Sharpe {f2(x['ev_v']['oos_sh'])} "
              f"(pos folds {x['ev_v']['n_pos']}/6).")
        if n == "H6":
            x = v["extra"]
            P(f"- Depth window: {x['kd']} folds kept {x['keep']} (id, months of depth training data), dropped {x['dropped']}; {x['months']} OOS months; "
              f"SE factor x{x['se']:.2f}; vetoed entry episodes {x['nveto']}; placebo OOS Sharpe {f2(x['e6p']['oos_sh'])}.")
        P("")
        P("Falsification tests:")
        P("")
        P(table(["criterion", "result", "met"], [(a, b, "yes" if c else "NO") for a, b, c in v["tests"]]))
        P("")

    # ---- combination step
    P("## S2 combination step")
    P("")
    elig = [n for n in names if V[n]["elig"]]
    if not elig:
        P("Not run: no hypothesis is eligible (S2.1 requires an individual WF pass AND the family DSR gate p > 0.95). "
          "`research/runs/combinations.json` is therefore not written and no combination trials are added to N.")
    else:
        P(f"Eligible components: {elig}. Combination execution is not implemented for this case in the script; enumerate manually per S2.")
    P("")

    # ---- S4 table
    best = max(names, key=lambda n: (V[n]["ev"]["stats"]["dsr"] if np.isfinite(V[n]["ev"]["stats"]["dsr"]) else -1))
    v = V[best]
    e = v["ev"]
    F = fams[v["fam"]]
    st = e["stats"]
    oos = F["oos"]
    cand = F["W"]["cand"][oos]
    bray_f = e["bray"]
    bray_pos = sum(1 for r in bray_f if r["sh"] > 0)
    beats_bray = sum(1 for r in e["rows"] if r["sh"] > r["bray"])
    bh_f = e["bh"]
    bh_pos = sum(1 for r in bh_f if r["sh"] > 0)
    P("## S4 success criteria - best candidate vs brayvb_v2 default and B&H")
    P("")
    first = "**Depth window applies (H6): S4 evaluated on the depth window.** " if best == "H6" else ""
    P(f"{first}Best candidate = highest DSR p among the seven variants: **{best}** ({'eligible' if v['elig'] else 'NOT eligible - shown for transparency only'}). "
      f"The holdout was never unlocked: holdout rows cannot be evaluated here (S2.6, only a final model passing everything earlier may run on it).")
    P("")
    deg = (e["is_sh"] - e["oos_sh"]) / e["is_sh"] if e["is_sh"] > 0 else float("nan")
    try:
        bb = block_bootstrap(cand, n_paths=4000, periods_per_year=PPY, seed=0)
        band = f"[{bb.pct('sharpe', 5):.2f}, {bb.pct('sharpe', 95):.2f}]"
    except Exception:
        band = "n/a"
    thr_wf = 4 if best != "H6" else meta["thr_d"]
    rows = [
        ("Deflated Sharpe p >= 0.95 (N per S2.4)", f"{f2(st['dsr'], 4)} (N={meta['N_with']:,})", "n/a", "n/a", "YES" if st["dsr"] >= 0.95 else "NO"),
        ("PBO < 0.15", f"{f2(st['pbo'], 3)}", "n/a (single strategy)", "n/a", "YES" if st["pbo"] < 0.15 else "NO"),
        (f"WF OOS Sharpe > 0 in >= {thr_wf}/{e['K']} folds", f"{e['n_pos']}/{e['K']}", f"{bray_pos}/{e['K']}", f"{bh_pos}/{e['K']}", "YES" if e["n_pos"] >= thr_wf else "NO"),
        (f"... and above brayvb_v2 default in >= {thr_wf}/{e['K']} folds", f"{beats_bray}/{e['K']}", "-", "-", "YES" if beats_bray >= thr_wf else "NO"),
        ("OOS/IS Sharpe degradation <= 50%", f"IS {f2(e['is_sh'])} -> OOS {f2(e['oos_sh'])} ({pct(deg, 0)})", "n/a", "n/a", "YES" if np.isfinite(deg) and deg <= 0.5 else "NO"),
        ("Pooled OOS Sharpe / return / max DD", f"{f2(e['oos_sh'])} / {pct(e['oos_ret'])} / {pct(e['oos_dd'])}",
         f"{f2(sh(bray_ret[oos]))} / {pct(tot_ret(bray_ret[oos]))} / {pct(maxdd(bray_ret[oos]))}",
         f"{f2(sh(bench[oos]))} / {pct(tot_ret(bench[oos]))} / {pct(maxdd(bench[oos]))}", "info"),
        ("Holdout net Sharpe >= B&H + 0.3 and >= best catalog", "not evaluated (holdout locked)", "-", "-", "n/a"),
        ("Holdout MaxDD <= 0.6 x B&H MaxDD", "not evaluated", "-", "-", "n/a"),
        ("Holdout inside 5-95% bootstrap band of WF OOS Sharpe", f"band {band} prepared; holdout not evaluated", "-", "-", "n/a"),
    ]
    P(table(["criterion", f"{best} (candidate)", "brayvb_v2 default", "B&H", "met by candidate"], rows))
    P("")
    P("**Verdict:** " + ("a candidate meets every pre-holdout criterion and may proceed to the single holdout run."
                         if (v["elig"] and st["pbo"] < 0.15 and beats_bray >= thr_wf and np.isfinite(deg) and deg <= 0.5)
                         else "no candidate meets the pre-holdout S4 criteria. The custom model is NOT better than the catalog strategy; the catalog strategy is preferred and nothing goes to the holdout or to live."))
    P("")
    P(f"_Runtime {time.time() - t0:.0f} s._")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    log("report written", REPORT)


if __name__ == "__main__":
    main()
