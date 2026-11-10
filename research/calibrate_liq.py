"""Calibrate the gross-opening liquidation model against OBSERVED Gate.io liquidations (BTCUSDT).

Free parameters (4): kappa (turnover->gross opening), half-life, leverage preset (categorical), global scale.
Scale is closed form (model USD is linear in scale): scale = sum(obs)/sum(model) on train.
Fit window: first 120 d of Gate history (scoring starts after a 30 d warm-up); eval: last 60 d.
Objective for the grid: Pearson corr of log1p(USD) per 1h bar (both sides summed), train only.
Usage: .venv/Scripts/python -m research.calibrate_liq   (threads=1; writes btclab/liquidity/liq_calib.json)
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btclab.data import loaders  # noqa: E402
from btclab.liquidity import liq_map  # noqa: E402
from research import validate_liq_model as V  # noqa: E402
from src.data import gate  # noqa: E402

SYM = "BTCUSDT"
KAPPAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
HALF_LIVES = [6, 24, 72, 240, 720]
PRESETS = {
    "default(10/25/50/100)": {10: .30, 25: .35, 50: .25, 100: .10},
    "low(3/5/10/25/50/100)": {3: .05, 5: .10, 10: .25, 25: .30, 50: .22, 100: .08},
    "mid(5/10/25/50)": {5: .15, 10: .30, 25: .35, 50: .20},
}
TRAIN_D, WARM_D = 120, 30


def load():
    g = gate.load(SYM)
    k = V.klines(SYM, g.index.min(), g.index.max() + pd.Timedelta("5min"))
    idx = g.index.intersection(k.index)
    g, k = g.loc[idx], k.loc[idx]
    oi = (g["open_interest_usd"] / g["mark_price"].where(g["mark_price"] > 0)).ffill(limit=12)
    mb = pd.read_parquet(loaders._base(SYM) / "um_futures" / "metrics_archive_5m.parquet", columns=["sum_open_interest"])
    mb.index = mb.index.tz_convert("UTC") if mb.index.tz else mb.index.tz_localize("UTC")
    oib = mb["sum_open_interest"].reindex(idx).ffill(limit=12)
    turn = liq_map.venue_turnover(k["volume"] * k["close"], oi, oib, 300.0)  # gate share of Binance turnover
    return idx, g, k, oi, turn


def run(k, oi, turn, kappa, hl, w, scale=1.0):
    lo, hi = float(k["low"].min()), float(k["high"].max())
    N = 700
    dl = (np.log(hi / lo) + 0.1) / N
    _, _, _, ev, _ = liq_map.liq_columns(k["close"], k["high"], k["low"], oi, 300.0, (np.log(lo) - 0.05, dl, N), 0,
                                         weights=w, turnover=turn, kappa=kappa, half_life_h=hl, scale=scale)
    c = np.zeros(len(k))
    for t, _s, usd, _p in ev:
        c[t] += usd
    return c


def stats(idx, obs, mod, a, b):
    """corr / ratio on [a,b) bar slice at 5m and 1h."""
    out = {}
    s5o = pd.Series(obs[a:b], idx[a:b]); s5m = pd.Series(mod[a:b], idx[a:b])
    for name, o, m in (("5m", s5o, s5m), ("1h", s5o.resample("1h").sum(), s5m.resample("1h").sum())):
        out[name] = dict(pearson_log=float(np.corrcoef(np.log1p(o), np.log1p(m))[0, 1]),
                         spearman=float(o.corr(m, method="spearman")),
                         ratio=float(m.sum() / o.sum()) if o.sum() else np.nan)
    return out


def main():
    idx, g, k, oi, turn = load()
    obs = (g["long_liq_usd"] + g["short_liq_usd"]).to_numpy()
    T = len(idx)
    nT, nW = TRAIN_D * 288, WARM_D * 288
    print("bars", T, "train", nT, flush=True)
    res = []
    for (pn, w), kap, hl in itertools.product(PRESETS.items(), KAPPAS, HALF_LIVES):
        m = run(k.iloc[:nT], oi.iloc[:nT], turn.iloc[:nT], kap, hl, w)   # train-only run (causal)
        st = stats(idx, obs, np.r_[m, np.zeros(T - nT)], nW, nT)
        sc = float(obs[nW:nT].sum() / max(m[nW:nT].sum(), 1.0))
        res.append(dict(preset=pn, kappa=kap, hl=hl, scale=sc, corr=st["1h"]["pearson_log"], sp=st["1h"]["spearman"]))
        print(res[-1], flush=True)
    df = pd.DataFrame(res).sort_values("corr", ascending=False)
    best = df.iloc[0]
    w = PRESETS[best.preset]
    # full run with chosen params and scale=1, slice into train / eval (state carries over, causal)
    m = run(k, oi, turn, best.kappa, best.hl, w)
    scale = float(obs[nW:nT].sum() / m[nW:nT].sum())
    mod = m * scale
    fit, ev = stats(idx, obs, mod, nW, nT), stats(idx, obs, mod, nT, T)
    # legacy baseline (kappa=0, hl=720, default weights, scale fit on train) for the same slices
    m0 = run(k, oi, turn, 0.0, 720, PRESETS["default(10/25/50/100)"])
    sc0 = float(obs[nW:nT].sum() / m0[nW:nT].sum())
    base_fit, base_ev = stats(idx, obs, m0 * sc0, nW, nT), stats(idx, obs, m0 * sc0, nT, T)
    base_raw = dict(fit_ratio=float(m0[nW:nT].sum() / obs[nW:nT].sum()), eval_ratio=float(m0[nT:].sum() / obs[nT:].sum()))
    out = dict(kappa=float(best.kappa), half_life_h=float(best.hl), scale=scale, preset=best.preset,
               weights={str(kk): v for kk, v in w.items()},
               fit=fit, eval=ev, legacy_fit=base_fit, legacy_eval=base_ev, legacy_raw_ratio=base_raw,
               gate_share_median=float((oi * k["close"] / g["open_interest_usd"]).median()),
               train_days=TRAIN_D, warm_days=WARM_D, bars=T, generated=str(pd.Timestamp.now("UTC")))
    (ROOT / "research" / "reports" / "liq_calib_fit.json").write_text(json.dumps(out, indent=1))
    (ROOT / "research" / "reports" / "liq_calib_grid.csv").write_text(df.to_csv(index=False))
    (ROOT / "btclab" / "liquidity" / "liq_calib.json").write_text(json.dumps(
        dict(kappa=out["kappa"], half_life_h=out["half_life_h"], scale=scale, weights=out["weights"], venue="gate",
             preset=out["preset"]), indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
