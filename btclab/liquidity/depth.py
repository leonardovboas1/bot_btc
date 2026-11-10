"""Binance bookDepth: snapshot pivot, heatmap grid, depth imbalance aligned to perp bars.

Archive schema: index=timestamp (UTC), percentage in {-5..-1,1..5}, depth (BTC), notional (USD,
cumulative from mid out to that band). Observations are used at their snapshot time (A1.4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import RAW, asof_series, bin_index, bin_center, tf_delta

BANDS = (1, 2, 3, 4, 5)


def load_depth(path=None, start=None, end=None) -> pd.DataFrame:
    df = pd.read_parquet(path or RAW / "bookdepth_archive.parquet")
    if start is not None or end is not None:
        df = df.loc[start:end]
    return df


def depth_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide: one row per snapshot, cols bid_1..bid_5 / ask_1..ask_5 (cumulative notional)."""
    d = df.reset_index()
    tcol = d.columns[0]
    d["side"] = np.where(d["percentage"] < 0, "bid", "ask")
    d["band"] = d["percentage"].abs().astype(int)
    d["col"] = d["side"] + "_" + d["band"].astype(str)
    w = d.pivot_table(index=tcol, columns="col", values="notional", aggfunc="last").sort_index()
    cols = [f"{s}_{b}" for s in ("bid", "ask") for b in BANDS]
    return w.reindex(columns=cols)


def depth_imbalance(snap: pd.DataFrame, perp_index: pd.DatetimeIndex, tf: str = "1h",
                    band: int = 1, max_age: str = "12h", kind: str = "imbalance") -> pd.Series:
    """(bid-ask)/(bid+ask) within +-band %, as-of the bar CLOSE (open+tf), backward only.

    NaN when the last snapshot is older than max_age. kind='total' returns bid+ask (placebo)."""
    bid, ask = snap[f"bid_{band}"], snap[f"ask_{band}"]
    s = (bid - ask) / (bid + ask) if kind == "imbalance" else bid + ask
    close_t = perp_index + tf_delta(tf)
    return pd.Series(asof_series(s, close_t, max_age), index=perp_index, name=f"depth_{kind}_{band}")


def depth_heatmap(snap: pd.DataFrame, close: pd.Series, tf: str = "1h", price_step: float = 0.005):
    """Time bucket x price bucket -> notional resting in that price slice.

    Per bucket the median snapshot is used; incremental band notional (cum diff) is spread
    uniformly over the band's price range around the bucket's bar close. Returns
    dict(times, price_edges_idx, prices, bid, ask) with 2-D arrays (T x P)."""
    inc = {}
    for s in ("bid", "ask"):
        c = snap[[f"{s}_{b}" for b in BANDS]]
        inc[s] = c.diff(axis=1)
        inc[s].iloc[:, 0] = c.iloc[:, 0]
    med = {s: inc[s].resample(tf).median() for s in inc}
    times = med["bid"].index
    cl = pd.Series(close.reindex(times).to_numpy(), index=times)  # bar close of that bucket's bar
    lo_i = bin_index(close.min() * 0.94, price_step)
    hi_i = bin_index(close.max() * 1.06, price_step)
    P = int(hi_i - lo_i + 1)
    out = {s: np.zeros((len(times), P)) for s in ("bid", "ask")}
    for ti, (t, px) in enumerate(cl.items()):
        if not np.isfinite(px):
            continue
        for s, sign in (("bid", -1), ("ask", 1)):
            for b in BANDS:
                v = med[s].iloc[ti][f"{s}_{b}"]
                if not np.isfinite(v):
                    continue
                a, z = sorted((px * (1 + sign * (b - 1) / 100), px * (1 + sign * b / 100)))
                ia, iz = bin_index(a, price_step) - lo_i, bin_index(z, price_step) - lo_i
                ia, iz = max(ia, 0), min(iz, P - 1)
                if iz >= ia:
                    out[s][ti, ia:iz + 1] += v / (iz - ia + 1)
    return {"times": times, "price_lo_idx": int(lo_i), "price_step": price_step,
            "prices": bin_center(np.arange(lo_i, hi_i + 1), price_step), "bid": out["bid"], "ask": out["ask"]}


def heatmap_json(hm: dict, max_cells: int = 200_000) -> dict:
    """JSON-serializable view (bid/ask as nested lists, time-decimated to fit max_cells)."""
    T, P = hm["bid"].shape
    stride = max(1, int(np.ceil(T * P / max_cells)))
    sl = slice(None, None, stride)
    return {"times": [t.isoformat() for t in hm["times"][sl]], "prices": [float(p) for p in hm["prices"]],
            "bid": np.round(hm["bid"][sl], 0).tolist(), "ask": np.round(hm["ask"][sl], 0).tolist(),
            "unit": "USD notional"}
