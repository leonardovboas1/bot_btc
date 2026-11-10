"""Does OBSERVED liquidation USD (Gate 5m, ~180d) coincide with price touching the MODELED bands?

Model: btclab.liquidity.liq_map.liq_columns on Binance 5m klines + Gate OI (USD/mark -> coin), causal.
A "touch" at bar t = the model consumed levels on that side (long levels when the low trades through,
short levels when the high does). Observed spike = Gate long_liq_usd (short_liq_usd) >= q-quantile
of that symbol's side-series. Both sides are tested separately.

Metrics (+-W bars tolerance, default 1 bar):
  hit rate  = P(touch within +-W | spike)       base = P(touch within +-W) over all bars
  lift      = hit rate / base ; placebo = circularly shifted spike series (>= 1 day offsets, keeps
              clustering), p = share of placebo hit rates >= observed.
Also Spearman corr between modeled consumed USD and observed liq USD per bar.
Usage: .venv/Scripts/python -m research.validate_liq_model  (writes research/reports/liq_validation.md)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from btclab.liquidity import liq_map  # noqa: E402
from src.data import gate  # noqa: E402

REPORT = ROOT / "research" / "reports" / "liq_validation.md"
FAPI = "https://fapi.binance.com/fapi/v1/klines"
WARM_BARS = 288 * 5
N_BINS = 700
Q = 0.95
W = 1
N_PLACEBO = 1000
MIN_SPIKE_USD = 0.0
MIN_DISTANCE = 288  # placebo shift >= 1 day


def klines(symbol: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Binance 5m klines t0..t1, cached under data/raw/gate/<SYM>/binance_klines_5m.parquet."""
    p = gate.GATE_DIR / symbol / "binance_klines_5m.parquet"
    old = pd.read_parquet(p) if p.exists() else None
    cur = int((old.index.max() + pd.Timedelta("5min")).timestamp() * 1000) if old is not None else int(t0.timestamp() * 1000)
    rows = []
    while cur < t1.timestamp() * 1000:
        r = requests.get(FAPI, params=dict(symbol=symbol, interval="5m", startTime=cur, limit=1500), timeout=20)
        r.raise_for_status()
        d = r.json()
        if not d:
            break
        rows += d
        cur = int(d[-1][0]) + 300_000
        if len(d) < 1500:
            break
        time.sleep(0.2)
    if rows:
        a = np.array([[float(x[i]) for i in range(6)] for x in rows])
        new = pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close", "volume"],
                           index=pd.to_datetime(a[:, 0], unit="ms", utc=True))
        old = new if old is None else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
        p.parent.mkdir(parents=True, exist_ok=True)
        old.to_parquet(p)
    return old


def model_touches(symbol: str):
    g = gate.load(symbol)
    if g is None or len(g) < 5000:
        raise RuntimeError(f"no gate data for {symbol}")
    k = klines(symbol, g.index.min(), g.index.max() + pd.Timedelta("5min"))
    idx = g.index.intersection(k.index)
    g, k = g.loc[idx], k.loc[idx]
    oi = (g["open_interest_usd"] / g["mark_price"].where(g["mark_price"] > 0)).to_numpy()
    oi = pd.Series(oi, index=idx).ffill(limit=12).to_numpy()
    lo, hi = float(k["low"].min()), float(k["high"].max())
    pad = 0.05
    dl = (np.log(hi / lo) + 2 * pad) / N_BINS
    edges = (np.log(lo) - pad, dl, N_BINS)
    _, _, _, events, _ = liq_map.liq_columns(k["close"], k["high"], k["low"], oi, 300.0, edges, 0)
    T = len(idx)
    cons = {"long": np.zeros(T), "short": np.zeros(T)}
    for t, side, usd, _p in events:
        cons[side][t] += usd
    rng_pct = ((k['high'] - k['low']) / k['close']).to_numpy()
    return idx, g, cons, rng_pct


def hit_stats(spike: np.ndarray, touch: np.ndarray, rng, rng_pct=None) -> dict:
    ntouch = np.convolve(touch.astype(float), np.ones(2 * W + 1), "same") > 0
    n_sp = int(spike.sum())
    if n_sp == 0:
        return dict(n=0)
    hit = float(ntouch[spike].mean())
    base = float(ntouch.mean())
    idxs = np.flatnonzero(spike)
    T = len(spike)
    pl = []
    for _ in range(N_PLACEBO):
        sh = rng.integers(MIN_DISTANCE, T - MIN_DISTANCE)
        pl.append(ntouch[(idxs + sh) % T].mean())
    pl = np.array(pl)
    vm = np.nan
    if rng_pct is not None:     # volatility-matched null: touch rate among bars in the same bar-range decile
        dec = np.clip(np.searchsorted(np.quantile(rng_pct, np.linspace(0.1, 0.9, 9)), rng_pct), 0, 9)
        rate = np.array([ntouch[dec == d].mean() if (dec == d).any() else np.nan for d in range(10)])
        vm = float(np.nanmean(rate[dec[spike]]))
    return dict(n=n_sp, hit=hit, vmbase=vm, vlift=hit / vm if vm else np.nan, base=base, lift=hit / base if base else np.nan,
                placebo=float(pl.mean()), plift=hit / pl.mean() if pl.mean() else np.nan,
                p=float((np.sum(pl >= hit) + 1) / (N_PLACEBO + 1)))


def main():
    rng = np.random.default_rng(7)
    rows, notes = [], []
    for sym in gate.SYMBOLS:
        try:
            idx, g, cons, rp = model_touches(sym)
        except Exception as e:
            notes.append(f"{sym}: skipped ({type(e).__name__}: {e})")
            continue
        for side in ("long", "short"):
            obs = g[f"{side}_liq_usd"].to_numpy()
            c = cons[side]
            valid = np.arange(len(idx)) >= WARM_BARS
            pos = obs[valid][obs[valid] > 0]
            thr = max(np.quantile(obs[valid], Q), np.quantile(pos, 0.5) if len(pos) else 0)
            spike = (obs >= thr) & (obs > 0) & valid
            cv, ov = c[valid], obs[valid]
            for tname, touch in (("any touch", c > 0),
                                 ("big touch (top 25% USD)", c >= np.quantile(c[c > 0], 0.75) if (c > 0).any() else c > 0)):
                touch = touch & valid
                s = hit_stats(spike, touch, rng, rp)
                if s["n"]:
                    s.update(symbol=sym, side=side, touch=tname, thr=float(thr),
                             rho=float(pd.Series(cv).corr(pd.Series(ov), method="spearman")))
                    rows.append(s)
        print(sym, "done", flush=True)
    df = pd.DataFrame(rows)
    write(df, notes)


def write(df: pd.DataFrame, notes: list):
    L = ["# Liquidation model validation (observed Gate liquidations vs modeled bands)", "",
         f"Generated {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC by `research/validate_liq_model.py`.", "",
         "## Method", "",
         "- Data: Gate.io USDT-perp `contract_stats` 5m (<=180 d): OI USD and long/short liquidation USD. "
         "Prices: Binance 5m klines. Model OI input = Gate OI USD / mark (coin units), "
         "default leverage mix, causal `liq_columns`, first 5 days discarded as warm-up.",
         f"- Spike = bar where observed liquidation USD on a side is >= the {int(Q*100)}th percentile of that symbol/side "
         "(and above the median positive value). Long and short tested separately: long spike vs modeled LONG levels consumed.",
         f"- Touch = model consumed levels of that side in the bar. Hit = touch within +-{W} bar of the spike. "
         "'big touch' = consumed USD in the top quartile of touch bars.",
         "- Lift = hit rate / unconditional touch rate. Placebo: spike series circularly shifted "
         f"{N_PLACEBO}x by >=1 day (preserves clustering); `plift` = hit / placebo mean; p = share of placebo >= observed.",
         "- Caveat: Gate is one venue (a fraction of market liquidations); a touch of a level in a 5m high/low is "
         "mechanically correlated with volatility, and spikes happen in volatile bars, so raw/placebo lift mostly reflects "
         "volatility. The vol-matched null compares the hit rate with the touch rate among bars in the same bar-range "
         "decile (high-low)/close as the spikes; it is the honest measure of model-specific information.", ""]
    if len(df):
        for touch in df["touch"].unique():
            L += [f"## {touch}", "", "| symbol | side | spikes | hit | base | lift | lift vs placebo | p | vol-matched base | vol-matched lift | spearman |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in df[df["touch"] == touch].itertuples():
                L.append(f"| {r.symbol} | {r.side} | {r.n} | {r.hit:.1%} | {r.base:.1%} | {r.lift:.2f} | "
                         f"{r.plift:.2f} | {r.p:.3f} | {r.vmbase:.1%} | {r.vlift:.2f} | {r.rho:.3f} |")
            a = df[df["touch"] == touch]
            L += ["", f"Pooled (spike-weighted): hit {np.average(a.hit, weights=a.n):.1%}, "
                      f"lift {np.average(a.lift, weights=a.n):.2f}, lift vs placebo {np.average(a.plift, weights=a.n):.2f}, "
                      f"volatility-matched lift {np.average(a.vlift, weights=a.n):.2f}; "
                      f"{int((a.p < 0.05).sum())}/{len(a)} symbol-sides with p<0.05.", ""]
    L += ["## Verdict", "", "_see summary printed below; filled automatically_", ""]
    if len(df):
        a = df[df["touch"] == "any touch"]
        pl = np.average(a.plift, weights=a.n)
        vl = np.average(a.vlift, weights=a.n)
        sig = int((a.p < 0.05).sum())
        v = (f"Versus shuffled-time placebo the model shows large lift (pooled {pl:.2f}, {sig}/{len(a)} symbol-sides p<0.05), "
             f"but that is mostly volatility: liquidation spikes occur in wide-range bars, which also cross more levels. "
             f"Against a volatility-matched null the pooled lift is {vl:.2f}. ")
        v += ("Only that last number measures model-specific information; " +
              ("it is close to 1, i.e. no evidence the modeled levels add information beyond volatility."
               if vl < 1.3 else "it is above 1, so the levels carry some information beyond volatility."))
        L[-2:] = [v, ""]
    if notes:
        L += ["## Notes", ""] + [f"- {n}" for n in notes]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-14:]))


if __name__ == "__main__":
    main()
