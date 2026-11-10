"""H7 data side (preregistration Amendment 3): causal 1h feature table, primary events, triple-barrier labels.

Conventions
-----------
* Row index = bar *open* time ``t``; every feature is known at the bar close ``t + 1h`` (the bar is
  complete) and uses nothing later.  Entry for an event at row ``t`` is the open of the next bar.
* Metrics (5 min archive) are observed at ``create_time + 5 min`` (A1.4); hourly values use only rows
  observed inside the bar (<= close).  No interpolation anywhere; NaN means "no signal".
* Holdout: data are loaded with ``end = cutoff`` (exclusive); the lock is never opened.
* Liquidation-map features come from a streaming re-implementation of the
  ``btclab.liquidity.liq_model.liq_levels`` state machine (same constants/decay/clearing), on a fixed
  absolute price-bin grid so that nothing depends on the data range, keeping only per-bar summaries.
* Events: one row per (bar, trigger).  Triggers / fixed direction rules:
    taker : |taker_z24| > 2          -> with the flow     (sign of z)
    basis : |basis_z7d| > 2          -> with the basis    (sign of z)
    oi    : |doi4_z| > 2             -> fade the 4h price move (-sign(ret_4h)); covers flush and surge
    liq   : modeled cluster consumed -> fade (long cluster consumed => long; short cluster => short;
                                       both => fade the bar's direction)
* Label (per h in 2, 4, 6): entry = open of bar t+1, TP = entry + dir*1.5*ATR24(t), SL = entry - dir*ATR24(t),
  vertical exit at the close of bar t+h.  Both barriers in one bar => SL (conservative).  y = 1 iff TP first.
  ``end_h`` = time at which the label is resolved (for purging).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from btclab.data import loaders
from btclab.liquidity._util import LOG_STEP, asof_series, bin_center, bin_index, rolling_z
from btclab.liquidity.liq_model import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "features"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
HS = (2, 4, 6)
LAG = pd.Timedelta("5min")
HOUR = pd.Timedelta("1h")
WARM = pd.Timedelta("31D")            # features start this long after the first metrics row
Z_TH = 2.0
TP_ATR, SL_ATR = 1.5, 1.0
CLUSTER_FRAC = 0.25                   # a bin is a "cluster" if >= 25 % of the side's largest bin

TRIGGERS = ("taker", "basis", "oi", "liq")


# ----------------------------------------------------------------------------- helpers
def _metrics_path(symbol: str) -> Path:
    return loaders._base(symbol) / "um_futures" / "metrics_archive_5m.parquet"


def load_metrics(symbol: str, end=None) -> pd.DataFrame:
    """Raw 5m metrics archive, cut at ``end`` (exclusive, by create_time); holdout cutoff by default."""
    m = pd.read_parquet(_metrics_path(symbol))
    end = loaders.get_cutoff() if end is None else end
    return m[m.index < loaders._utc(end)]


def _asof(obs: pd.Series, index: pd.DatetimeIndex, lag=LAG, max_age="12h") -> np.ndarray:
    """Value of ``obs`` last published (observation time + lag) at or before the bar close, max age 12h."""
    o = obs.copy()
    o.index = o.index + lag
    return asof_series(o, index + HOUR, max_age)


def atr(perp: pd.DataFrame, n: int = 24) -> pd.Series:
    pc = perp["close"].shift(1)
    tr = pd.concat([perp["high"] - perp["low"], (perp["high"] - pc).abs(), (perp["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def liq_summary(close, high, low, oi, atr_, weights=None, mmr=0.004, decay=0.999, long_share=0.5,
                step=LOG_STEP, warm: int = 720) -> pd.DataFrame:
    """Per-bar liquidation-map summary from one causal streaming pass (same model as liq_model.liq_levels).

    Row t uses bars <= t only.  Columns: liq_long_dist_atr, liq_short_dist_atr (distance to the nearest
    modeled cluster below / above close, in ATR), liq_long_size, liq_short_size (cluster mass / OI notional),
    liq_long_consumed, liq_short_consumed (a cluster bin was cleared by this bar's range).
    """
    weights = weights or DEFAULT_WEIGHTS
    close, high, low, oi, atr_ = (np.asarray(a, float) for a in (close, high, low, oi, atr_))
    T = len(close)
    lo_i = int(bin_index(0.1, step)); hi_i = int(bin_index(1e6, step)); P = hi_i - lo_i + 1
    L = np.zeros(P); S = np.zeros(P)
    ws = np.array(list(weights.values()), float); ws /= ws.sum()
    levs = np.array(list(weights.keys()), float)
    out = {k: np.full(T, np.nan) for k in ("liq_long_dist_atr", "liq_short_dist_atr", "liq_long_size", "liq_short_size")}
    cons_l = np.zeros(T, bool); cons_s = np.zeros(T, bool)
    prev_oi = np.nan
    first_valid = None
    for t in range(T):
        L *= decay; S *= decay
        if np.isfinite(oi[t]) and np.isfinite(prev_oi) and prev_oi > 0 and oi[t] < prev_oi:
            f = oi[t] / prev_oi
            L *= f; S *= f
        il = int(bin_index(low[t], step)) - lo_i
        ih = int(bin_index(high[t], step)) - lo_i
        ml, ms = L.max(), S.max()
        if ml > 0 and L[max(il, 0):].max() >= CLUSTER_FRAC * ml:
            cons_l[t] = True
        if ms > 0 and S[: max(ih + 1, 0)].max() >= CLUSTER_FRAC * ms:
            cons_s[t] = True
        L[max(il, 0):] = 0.0
        S[: max(ih + 1, 0)] = 0.0
        if np.isfinite(oi[t]):
            if first_valid is None:
                first_valid = t
            if np.isfinite(prev_oi) and oi[t] > prev_oi:
                notional = (oi[t] - prev_oi) * close[t]
                for lv, w in zip(levs, ws):
                    pl = close[t] * (1 - 1 / lv + mmr)
                    ps = close[t] * (1 + 1 / lv - mmr)
                    if pl > 0:
                        i = int(bin_index(pl, step)) - lo_i
                        if 0 <= i < P:
                            L[i] += notional * long_share * w
                    i = int(bin_index(ps, step)) - lo_i
                    if 0 <= i < P:
                        S[i] += notional * (1 - long_share) * w
            prev_oi = oi[t]
        # summary after the update (bars <= t)
        ic = max(int(bin_index(close[t], step)) - lo_i, 0)
        a = atr_[t]
        onot = oi[t] * close[t] if np.isfinite(oi[t]) else np.nan
        mxl, mxs = L.max(), S.max()
        if mxl > 0 and np.isfinite(a) and a > 0:
            idx = np.flatnonzero(L[: ic + 1] >= CLUSTER_FRAC * mxl)
            if len(idx):
                j = idx[-1]
                out["liq_long_dist_atr"][t] = (close[t] - bin_center(j + lo_i, step)) / a
                out["liq_long_size"][t] = L[j] / onot if onot > 0 else np.nan
        if mxs > 0 and np.isfinite(a) and a > 0:
            idx = np.flatnonzero(S[ic:] >= CLUSTER_FRAC * mxs)
            if len(idx):
                j = idx[0] + ic
                out["liq_short_dist_atr"][t] = (bin_center(j + lo_i, step) - close[t]) / a
                out["liq_short_size"][t] = S[j] / onot if onot > 0 else np.nan
    df = pd.DataFrame(out)
    df["liq_long_consumed"] = cons_l.astype(float)
    df["liq_short_consumed"] = cons_s.astype(float)
    if first_valid is None:
        df[:] = np.nan
    else:  # model state needs a warm-up after the first OI observation
        df.iloc[: first_valid + warm] = np.nan
    return df


# ----------------------------------------------------------------------------- features
def build_features(perp: pd.DataFrame, spot_close: pd.Series, funding: pd.Series, metrics: pd.DataFrame,
                   symbol: str = "BTCUSDT", start=None) -> pd.DataFrame:
    """Causal 1h feature table indexed by bar open time (known at close).  See module docstring."""
    idx = perp.index
    f = pd.DataFrame(index=idx)
    close, high, low, opn = perp["close"], perp["high"], perp["low"], perp["open"]
    f["close"] = close
    f["atr24"] = atr(perp)
    f["atr_pct"] = f["atr24"] / close

    # --- 1. order flow
    obs = metrics.index + LAG
    ratio = metrics["sum_taker_long_short_vol_ratio"]
    lr = np.log(ratio.where(ratio > 0))
    g = pd.DataFrame({"lr": lr.to_numpy()}, index=obs)
    key = (obs - pd.Timedelta("1ns")).floor("h")                 # bar open that contains the observation
    agg = g.groupby(key)["lr"].agg(["mean", "count"])
    tk = agg["mean"].where(agg["count"] >= 6).reindex(idx)      # >= 6 of 12 five-minute rows, else NaN
    f["taker_lr"] = tk.to_numpy()
    f["taker_z24"] = rolling_z(tk.to_numpy(), 24)
    f["taker_z168"] = rolling_z(tk.to_numpy(), 168)
    net = 2 * perp["taker_buy_base"] - perp["volume"]
    f["taker_burst"] = (net / perp["volume"].rolling(168, min_periods=134).mean()).to_numpy()

    # --- 3. derivatives positioning
    oi = pd.Series(_asof(metrics["sum_open_interest"], idx), index=idx).where(lambda s: s > 0)
    for k in (1, 4, 24):
        d = np.log(oi / oi.shift(k))
        f[f"doi{k}_z"] = rolling_z(d.to_numpy(), 720)
    fz = pd.Series(rolling_z(funding.to_numpy(), 90), index=funding.index)   # on 8h prints (30 d)
    f["funding_z30d"] = _asof(fz, idx, lag=pd.Timedelta("1min"))  # print usable 1 min after settlement
    for col, name in (("count_long_short_ratio", "ls_global_z"), ("sum_toptrader_long_short_ratio", "ls_top_z")):
        s = pd.Series(_asof(metrics[col], idx), index=idx)
        f[name] = rolling_z(s.to_numpy(), 720)

    # --- 2. liquidation map
    ls = liq_summary(close.to_numpy(), high.to_numpy(), low.to_numpy(), oi.to_numpy(), f["atr24"].to_numpy())
    for c in ls.columns:
        f[c] = ls[c].to_numpy()

    # --- 4. basis
    sp = spot_close.reindex(idx)
    basis = (close - sp) / sp
    f["basis"] = basis
    f["basis_z7d"] = rolling_z(basis.to_numpy(), 168)
    f["basis_chg4h"] = basis - basis.shift(4)

    # --- 5. session (bar open time, UTC)
    hr = idx.hour.to_numpy(); wd = idx.dayofweek.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24); f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["wd_sin"] = np.sin(2 * np.pi * wd / 7); f["wd_cos"] = np.cos(2 * np.pi * wd / 7)
    f["sess_asia"] = ((hr >= 0) & (hr < 8)).astype(float)
    f["sess_europe"] = ((hr >= 8) & (hr < 16)).astype(float)
    f["sess_us"] = (hr >= 16).astype(float)

    # --- context
    r1 = np.log(close / close.shift(1))
    f["rvol24"] = r1.rolling(24, min_periods=24).std(ddof=0)
    f["rvol_z"] = rolling_z(f["rvol24"].to_numpy(), 720)
    f["ret_4h"] = np.log(close / close.shift(4))
    f["ret_24h"] = np.log(close / close.shift(24))
    path = close.diff().abs().rolling(24, min_periods=24).sum()
    f["er24"] = (close - close.shift(24)).abs() / path.where(path > 0)
    f["bar_dir"] = np.sign(close - opn)
    coin = symbol.upper().replace("USDT", "")
    for c in (s.replace("USDT", "") for s in SYMBOLS):
        f[f"coin_{c}"] = float(c == coin)

    # flags (A1.4.6)
    f["metrics_gap"] = (f["taker_lr"].isna() | oi.isna()).astype(bool)
    f["spot_gap"] = sp.isna().to_numpy()
    start = (metrics.index.min() + WARM).ceil("h") if start is None else loaders._utc(start)
    return f[f.index >= start]


# ----------------------------------------------------------------------------- events + labels
def make_events(f: pd.DataFrame, perp: pd.DataFrame, hs=HS, z_th: float = Z_TH, symbol: str = "") -> pd.DataFrame:
    """One row per (bar, trigger) with fixed direction and triple-barrier labels for every h."""
    def sgn(x):
        return np.sign(x.to_numpy())

    parts = []
    tz = f["taker_z24"]
    parts.append(("taker", (tz.abs() > z_th).to_numpy(), sgn(tz)))
    bz = f["basis_z7d"]
    parts.append(("basis", (bz.abs() > z_th).to_numpy(), sgn(bz)))
    oz = f["doi4_z"]
    parts.append(("oi", (oz.abs() > z_th).to_numpy(), -sgn(f["ret_4h"])))
    cl, cs = f["liq_long_consumed"].to_numpy() == 1, f["liq_short_consumed"].to_numpy() == 1
    d = np.where(cl & ~cs, 1.0, np.where(cs & ~cl, -1.0, np.where(cl & cs, -sgn(f["bar_dir"]), 0.0)))
    parts.append(("liq", cl | cs, d))

    pidx = perp.index
    pos_all = pidx.get_indexer(f.index)
    O, H, Lw, C = (perp[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    N = len(perp)
    atr_all = f["atr24"].to_numpy()
    rows = []
    for name, mask, dirn in parts:
        sel = mask & np.isfinite(dirn) & (dirn != 0) & np.isfinite(atr_all)
        if sel.any():
            rows.append(pd.DataFrame({"bar_time": f.index[sel], "trigger": name,
                                      "direction": dirn[sel].astype(int), "pos": pos_all[sel],
                                      "atr": atr_all[sel]}))
    if not rows:
        return pd.DataFrame()
    ev = pd.concat(rows, ignore_index=True).sort_values(["bar_time", "trigger"], ignore_index=True)
    pos = ev["pos"].to_numpy()
    dirn = ev["direction"].to_numpy().astype(float)
    atr_v = ev["atr"].to_numpy()
    j = pos + 1                                              # entry bar
    ok_entry = j < N
    jj = np.minimum(j, N - 1)
    entry = O[jj]
    ev["entry_time"] = ev["bar_time"] + HOUR
    ev["entry_price"] = np.where(ok_entry, entry, np.nan)
    tp = entry + dirn * TP_ATR * atr_v
    sl = entry - dirn * SL_ATR * atr_v
    ts = pidx.asi8
    ar = np.arange(len(ev))
    for h in hs:
        w = np.minimum(jj[:, None] + np.arange(h)[None, :], N - 1)
        okw = ok_entry & (j + h - 1 < N) & ((ts[w[:, -1]] - ts[w[:, 0]]) == (h - 1) * HOUR.value)
        hi, lo = H[w], Lw[w]
        up = (dirn > 0)[:, None]
        tp_hit = np.where(up, hi >= tp[:, None], lo <= tp[:, None])
        sl_hit = np.where(up, lo <= sl[:, None], hi >= sl[:, None])
        big = h + 1
        tp_i = np.where(tp_hit.any(1), tp_hit.argmax(1), big)
        sl_i = np.where(sl_hit.any(1), sl_hit.argmax(1), big)
        outcome = np.where((sl_i == big) & (tp_i == big), 0, np.where(sl_i <= tp_i, -1, 1))  # SL wins ties
        exit_i = np.where(outcome == 1, tp_i, np.where(outcome == -1, sl_i, h - 1))
        exit_px = np.where(outcome == 1, tp, np.where(outcome == -1, sl, C[w[:, -1]]))
        end_t = pd.to_datetime(ts[w[ar, exit_i]] + HOUR.value, utc=True).as_unit("ns")
        ev[f"y_{h}"] = np.where(okw, (outcome == 1).astype(float), np.nan)
        ev[f"outcome_{h}"] = np.where(okw, outcome, np.nan)
        ev[f"ret_atr_{h}"] = np.where(okw, dirn * (exit_px - entry) / atr_v, np.nan)
        ev[f"end_{h}"] = pd.Series(end_t).where(okw, pd.NaT).to_numpy()
    ev["symbol"] = symbol
    return ev.drop(columns="pos")


# ----------------------------------------------------------------------------- builder
def build_symbol(symbol: str, out_dir: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = loaders.get_cutoff()
    metrics = load_metrics(symbol, cutoff)
    t0 = metrics.index.min().floor("h")
    perp = loaders.load_ohlcv("perp", "1h", start=t0, end=cutoff, symbol=symbol)
    spot = loaders.load_ohlcv("spot", "1h", end=cutoff, symbol=symbol)["close"]
    fund = loaders.load_funding(end=cutoff, symbol=symbol)
    feat = build_features(perp, spot, fund, metrics, symbol)
    ev = make_events(feat, perp, symbol=symbol)
    feat.insert(0, "symbol", symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_dir / f"h7_{symbol}_1h.parquet")
    ev.to_parquet(out_dir / f"h7_{symbol}_1h_events.parquet")
    return feat, ev


FEATURE_COLS = ["taker_z24", "taker_z168", "taker_burst", "liq_long_dist_atr", "liq_short_dist_atr", "liq_long_size",
                "liq_short_size", "liq_long_consumed", "liq_short_consumed", "doi1_z", "doi4_z", "doi24_z",
                "funding_z30d", "ls_global_z", "ls_top_z", "basis_z7d", "basis_chg4h", "hour_sin", "hour_cos",
                "wd_sin", "wd_cos", "sess_asia", "sess_europe", "sess_us", "rvol_z", "ret_24h", "er24"]


def write_report(symbols, out_dir: Path = OUT, path: Path | None = None) -> str:
    path = path or ROOT / "research" / "reports" / "h7_features.md"
    L = ["# H7 data report (features, events, labels)", "",
         "Data side only (Amendment 3). No strategy performance is computed here. Holdout cutoff "
         f"{loaders.get_cutoff().isoformat()} (exclusive) is enforced; no row, feature or label end time reaches it.",
         "", "## Rows and range", "",
         "| symbol | rows | first bar | last bar | metrics_gap | spot_gap |", "|---|---|---|---|---|---|"]
    data = {}
    for s in symbols:
        f = pd.read_parquet(out_dir / f"h7_{s}_1h.parquet")
        e = pd.read_parquet(out_dir / f"h7_{s}_1h_events.parquet")
        data[s] = (f, e)
        L.append(f"| {s} | {len(f)} | {f.index.min()} | {f.index.max()} | "
                 f"{f['metrics_gap'].mean():.2%} | {f['spot_gap'].mean():.2%} |")
    L += ["", "Alt metrics start 2021-12-01 (BTC 2021-01-01); tables start 31 days after the first metrics row "
          "(warm-up for 30 d z-scores and the liquidation model).", "",
          "## Events per trigger and direction (count, labelled count at h=4 in parentheses)", "",
          "| symbol | trigger | long | short |", "|---|---|---|---|"]
    for s, (f, e) in data.items():
        for t in TRIGGERS:
            x = e[e.trigger == t]
            cells = [f"{(x.direction == d).sum()} ({int(((x.direction == d) & x['y_4'].notna()).sum())})" for d in (1, -1)]
            L.append(f"| {s} | {t} | {cells[0]} | {cells[1]} |")
    L += ["", "## Label balance (share of events by first barrier hit; labelled events only)", "",
          "| symbol | " + " | ".join(f"h={h} TP / SL / vertical" for h in HS) + " |", "|---|" + "---|" * len(HS)]
    for s, (f, e) in data.items():
        cells = []
        for h in HS:
            o = e[f"outcome_{h}"].dropna()
            cells.append(f"{(o == 1).mean():.1%} / {(o == -1).mean():.1%} / {(o == 0).mean():.1%} (n={len(o)})")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "## NaN fraction per feature", "", "| feature | " + " | ".join(symbols) + " |", "|---|" + "---|" * len(symbols)]
    for c in FEATURE_COLS:
        L.append(f"| {c} | " + " | ".join(f"{data[s][0][c].isna().mean():.1%}" for s in symbols) + " |")
    L += ["", "## Notes", "",
          "- Features are known at bar close (index = bar open); entry is the next bar's open; barriers use real prices.",
          "- `liq_*` features are a modeled map (OI-delta heuristic), not observed liquidations.",
          "- A bar can appear under several triggers (one row per bar and trigger); events lacking h future bars are unlabelled (NaN).",
          "- NaN is kept as 'no signal' (A1.4); no interpolation or imputation was applied.", ""]
    txt = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build H7 feature/event/label tables")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--report", action="store_true", help="write research/reports/h7_features.md afterwards")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    for s in a.symbols:
        f, e = build_symbol(s, out)
        print(f"{s}: {len(f)} rows {f.index.min()} -> {f.index.max()}, {len(e)} events", flush=True)
    if a.report:
        write_report(a.symbols, out)


if __name__ == "__main__":
    main()
