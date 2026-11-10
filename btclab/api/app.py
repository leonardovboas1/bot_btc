"""FastAPI app. Backtest stats use in-sample data only (bars < holdout cutoff)."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from btclab.api.presets import PRESETS, get_preset
from btclab.data import loaders
from btclab.engine.costs import PRESETS as COST_PRESETS
from btclab.engine.sim import EXIT_REASON_NAMES
from btclab.liquidity import service as liq_service
from btclab.liquidity._util import RAW as LIQ_RAW
from btclab.indicators.registry import REGISTRY, compute
from btclab.signals import run_spec
from btclab.signals.dsl import SpecError

app = FastAPI(title="BTC Signal Lab API")
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

GREEN, RED, GREY = "#26a69a", "#ef5350", "#9e9e9e"


# ------------------------------------------------------------------ data cache
@lru_cache(maxsize=32)
def _full(market: str, tf: str) -> pd.DataFrame:
    """All bars incl. holdout (treat as read-only); perp gets a funding column."""
    try:
        df = loaders.load_ohlcv(market, tf, allow_holdout=True)
        if market == "perp":
            df = df.copy()
            df["funding"] = loaders.load_funding(tf, allow_holdout=True).values
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return df


def _cutoff() -> pd.Timestamp:
    return loaders.get_cutoff()


def _ts(idx) -> np.ndarray:
    return (pd.DatetimeIndex(idx).asi8 // 10**9).astype(np.int64)


def _num(x):
    """Make JSON-safe: NaN/inf -> None, numpy ints -> int."""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    return x


def _resolve_spec(strategy: Optional[str], spec: Optional[dict], tf: Optional[str] = None) -> dict:
    if spec is None:
        if strategy not in PRESETS:
            raise HTTPException(404, f"unknown strategy {strategy!r}")
        spec = get_preset(strategy)
    spec = dict(spec)
    if tf:
        spec["tf"] = tf
    return spec


def _run(spec: dict, df: pd.DataFrame):
    try:
        return run_spec(spec, df)
    except (SpecError, ValueError, KeyError) as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


def _local(df, ts):
    return ts.tz_convert(df.index.tz) if df.index.tz is not None else ts.tz_convert("UTC").tz_localize(None)


# ------------------------------------------------------------------ endpoints
@app.get("/ohlcv")
def ohlcv(market: str = "perp", tf: str = "4h", start: Optional[str] = None,
          end: Optional[str] = None):
    _full(market, tf)
    df = loaders.load_ohlcv(market, tf, start, end, allow_holdout=True)
    t = _ts(df.index)
    bars = [{"time": int(a), "open": o, "high": h, "low": l, "close": c, "volume": v}
            for a, o, h, l, c, v in zip(t, df["open"], df["high"], df["low"],
                                        df["close"], df["volume"])]
    return {"market": market, "tf": tf, "holdout_start": int(_cutoff().timestamp()), "bars": bars}


@app.get("/strategies")
def strategies():
    return [get_preset(k) for k in PRESETS]


class BacktestReq(BaseModel):
    strategy: Optional[str] = None
    spec: Optional[dict] = None
    market: str = "perp"
    tf: Optional[str] = None
    fee: Optional[str] = None      # cost preset name, e.g. "binance_perp_taker"


def _markers(df, trades):
    idx = _ts(df.index)
    out = []
    for tr in trades:
        long = tr["side"] == 1
        out.append({"time": int(idx[tr["entry_idx"]]), "position": "belowBar" if long else "aboveBar",
                    "shape": "arrowUp" if long else "arrowDown",
                    "color": GREEN if long else RED, "text": "LONG" if long else "SHORT"})
        out.append({"time": int(idx[tr["exit_idx"]]), "position": "aboveBar" if long else "belowBar",
                    "shape": "circle", "color": GREY, "text": "EXIT"})
    return sorted(out, key=lambda m: m["time"])


def _trade_rows(df, trades):
    idx = _ts(df.index)
    return [{"entry_time": int(idx[t["entry_idx"]]), "exit_time": int(idx[t["exit_idx"]]),
             "side": "LONG" if t["side"] == 1 else "SHORT",
             "entry_price": float(t["entry_px"]), "exit_price": float(t["exit_px"]),
             "pnl_pct": float(t["pnl_pct"]),
             "exit_reason": EXIT_REASON_NAMES.get(int(t["exit_reason"]), str(int(t["exit_reason"])))}
            for t in trades]


@app.post("/backtest")
def backtest(req: BacktestReq):
    spec = _resolve_spec(req.strategy, req.spec, req.tf)
    if req.fee:
        spec["costs"] = req.fee
    tf = spec["tf"]
    full = _full(req.market, tf)
    n_is = int(full.index.searchsorted(_local(full, _cutoff())))
    if n_is < 50:
        raise HTTPException(400, "not enough in-sample data")
    ins = full.iloc[:n_is]
    res, metrics = _run(spec, ins)
    t = _ts(ins.index)
    eq = res.equity
    close = ins["close"].to_numpy(dtype=float)
    bh = eq[0] * close / close[0]

    # Holdout: signals only (run over the full series so state is continuous); no stats.
    hold_markers: list = []
    if len(full) > n_is:
        fres, _ = _run(spec, full)
        hold_markers = _markers(full, fres.trades[fres.trades["entry_idx"] >= n_is])
    return _num({
        "strategy": spec["name"], "market": req.market, "tf": tf,
        "holdout_start": int(_cutoff().timestamp()),
        "trades": _trade_rows(ins, res.trades),
        "markers": _markers(ins, res.trades),
        "equity": [{"time": int(a), "value": float(b)} for a, b in zip(t, eq)],
        "buy_hold": [{"time": int(a), "value": float(b)} for a, b in zip(t, bh)],
        "metrics": metrics,
        "holdout_markers": hold_markers,
    })


@app.get("/signal/current")
def signal_current(strategy: str = "brayvb_v2", tf: Optional[str] = None, market: str = "perp"):
    spec = _resolve_spec(strategy, None, tf)
    df = _full(market, spec["tf"])
    res, _ = _run(spec, df)
    pos = int(res.position[-1])
    return {"strategy": strategy, "tf": spec["tf"],
            "state": {1: "LONG", -1: "SHORT"}.get(pos, "FLAT"),
            "time": int(_ts(df.index[-1:])[0]), "price": float(df["close"].iloc[-1])}


@app.get("/indicator/{name}")
def indicator(name: str, params: str = Query("{}", description="JSON object"),
              tf: str = "4h", market: str = "perp"):
    if name not in REGISTRY:
        raise HTTPException(404, f"unknown indicator {name!r}")
    try:
        p = json.loads(params)
        if not isinstance(p, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "params must be a JSON object")
    df = _full(market, tf)
    try:
        out = compute(name, df, **p)
    except TypeError as e:
        raise HTTPException(400, str(e))
    t = _ts(df.index)
    series = {}
    for k, v in out.items():
        a = np.asarray(v)
        if a.ndim != 1 or len(a) != len(t):
            continue
        a = a.astype(float)
        ok = np.isfinite(a)
        series[k] = [{"time": int(x), "value": float(y)} for x, y in zip(t[ok], a[ok])]
    return {"name": name, "tf": tf, "series": series}


# ------------------------------------------------------------------ fees
@app.get("/fees")
def fees():
    return [m.to_dict() for m in COST_PRESETS.values()]


# ------------------------------------------------------------------ liquidity layers
MAX_COLS, MAX_ROWS = 400, 150
_TF_SEC = {"1h": 3600, "4h": 14400, "1d": 86400}


@lru_cache(maxsize=1)
def _book_range() -> tuple[int, int]:
    idx = pd.read_parquet(LIQ_RAW / "bookdepth_archive.parquet", columns=[]).index
    return int(idx.min().timestamp()), int(idx.max().timestamp())


def _groups(n: int, m: int) -> np.ndarray:
    """Start offsets that split n items into <= m near-equal groups."""
    k = min(n, m)
    return np.unique(np.linspace(0, n, k, endpoint=False).astype(int))


def _shrink(times, prices, grids: dict, cols: int, rows: int) -> dict:
    """Mean over merged time columns, sum over merged price rows (notional per slice)."""
    t = np.asarray([pd.Timestamp(x).timestamp() for x in times], dtype=float)
    p = np.asarray(prices, dtype=float)
    ci, ri = _groups(len(t), cols), _groups(len(p), rows)
    cn = np.diff(np.append(ci, len(t)))
    pn = np.diff(np.append(ri, len(p)))
    out = {"times": [int(x) for x in t[ci]], "prices": [round(float(x), 2) for x in np.add.reduceat(p, ri) / pn]}
    for k, g in grids.items():
        a = np.asarray(g, dtype=float).reshape(len(t), len(p))
        a = np.add.reduceat(a, ci, axis=0) / cn[:, None]
        a = np.add.reduceat(a, ri, axis=1)
        out[k] = np.round(a / 1e6, 3).tolist()  # USD millions
    return out


def _window(tf: str, start: Optional[str], end: Optional[str]) -> tuple[str, str]:
    if tf not in _TF_SEC:
        raise HTTPException(400, f"unsupported tf {tf!r}")
    e = pd.Timestamp(end) if end else pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    s = pd.Timestamp(start) if start else e - pd.Timedelta(seconds=_TF_SEC[tf] * MAX_COLS)
    if s >= e:
        raise HTTPException(400, "start must be before end")
    return str(s), str(e)


def _limit(cols: int, rows: int) -> tuple[int, int]:
    return max(1, min(cols, MAX_COLS)), max(1, min(rows, MAX_ROWS))


@app.get("/heatmap/book")
def heatmap_book(tf: str = "1h", start: Optional[str] = None, end: Optional[str] = None,
                 rows: int = MAX_ROWS, cols: int = MAX_COLS):
    s, e = _window(tf, start, end)
    lo, hi = _book_range()
    avail = {"available_from": lo, "available_to": hi}
    if int(pd.Timestamp(e, tz="UTC").timestamp()) < lo or int(pd.Timestamp(s, tz="UTC").timestamp()) > hi:
        return {"empty": True, **avail, "times": [], "prices": [], "bid": [], "ask": []}
    try:
        d = liq_service.depth_heatmap_payload(s, e, tf)
    except (FileNotFoundError, ValueError, KeyError) as ex:
        raise HTTPException(400, str(ex))
    if not d["times"]:
        return {"empty": True, **avail, "times": [], "prices": [], "bid": [], "ask": []}
    cols, rows = _limit(cols, rows)
    return {"empty": False, **avail, "unit": "USD milhões",
            **_shrink(d["times"], d["prices"], {"bid": d["bid"], "ask": d["ask"]}, cols, rows)}


@app.get("/heatmap/liq")
def heatmap_liq(tf: str = "1h", start: Optional[str] = None, end: Optional[str] = None,
                rows: int = MAX_ROWS, cols: int = MAX_COLS):
    s, e = _window(tf, start, end)
    try:
        d = liq_service.liq_payload(s, e, tf)
    except (FileNotFoundError, ValueError, KeyError) as ex:
        raise HTTPException(400, str(ex))
    if not d["times"]:
        return {"empty": True, "times": [], "prices": [], "long": [], "short": [], "flush_times": []}
    cols, rows = _limit(cols, rows)
    out = _shrink(d["times"], d["prices"], {"long": d["long"], "short": d["short"]}, cols, rows)
    out["flush_times"] = [int(pd.Timestamp(x).timestamp()) for x in d.get("flush_times", [])]
    return {"empty": False, "unit": "USD milhões (estimado)", **out}


@app.get("/volume_profile")
def volume_profile(tf: str = "1h", start: Optional[str] = None, end: Optional[str] = None,
                   rows: int = MAX_ROWS):
    s, e = _window(tf, start, end)
    try:
        d = liq_service.volume_profile_payload(s, e, tf)
    except (FileNotFoundError, ValueError, KeyError) as ex:
        raise HTTPException(400, str(ex))
    p = np.asarray(d["prices"], dtype=float)
    v = np.asarray(d["volume"], dtype=float)
    ri = _groups(len(p), max(1, min(rows, MAX_ROWS)))
    n = np.diff(np.append(ri, len(p)))
    out = {"prices": [round(float(x), 2) for x in np.add.reduceat(p, ri) / n],
           "volume": [round(float(x), 2) for x in np.add.reduceat(v, ri)]}
    for k in ("poc", "vah", "val"):
        if k in d:
            out[k] = d[k]
    return _num(out)
