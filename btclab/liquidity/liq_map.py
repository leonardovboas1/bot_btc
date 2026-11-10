"""Coinglass-style liquidation map: modeled from OI deltas, consumed on touch. NOT a forecast.

Each bar's OI increase (BTC) x close = new notional, split long/short and spread over a leverage
mix; an L-x long opened at close c liquidates near c*(1-1/L+mmr), a short near c*(1+1/L-mmr).
A level is removed the first time a later bar's high/low trades through it (consumption event).
Column t depends only on bars <= t (causal). A warm-up before the window start lets levels
created earlier but not yet consumed show up. Resolution depends on the range, not on the chart.
"""
from __future__ import annotations

import base64
import threading
import time

import numpy as np
import pandas as pd
import requests

import json
from pathlib import Path

from btclab.data import loaders
from .liq_model import DEFAULT_WEIGHTS

RANGES = {  # name: (window, resolution)
    "1d": ("1D", "5min"), "3d": ("3D", "5min"), "1w": ("7D", "15min"), "2w": ("14D", "15min"),
    "1m": ("30D", "1h"), "3m": ("90D", "1h"), "6m": ("180D", "1h"),
}
VENUES = ("binance", "bybit", "gate", "okx")
N_ROWS = 280
MMR = 0.004
HALF_LIFE_H = 24 * 30        # slow time-based decay of stale estimates
MAX_EVENTS = 600
FAPI = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
OKX = "https://www.okx.com"
TTL = 300.0
MIN_WARM = pd.Timedelta("30D")   # warm-up = max(30 d, 2 x window)
CALIB_PATH = Path(__file__).with_name("liq_calib.json")


def load_calib() -> dict:
    """Fitted parameters (research/calibrate_liq.py). Missing file -> legacy net-dOI behavior."""
    try:
        c = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        c["weights"] = {float(k): float(v) for k, v in c["weights"].items()}
        return c
    except Exception:
        return dict(kappa=0.0, half_life_h=HALF_LIFE_H, scale=1.0, weights=dict(DEFAULT_WEIGHTS))


CALIB = load_calib()

_rest_cache: dict = {}
_rest_lock = threading.Lock()
_file_cache: dict = {}


def _get(url: str, params: dict) -> object:
    key = (url, tuple(sorted(params.items())))
    now = time.time()
    with _rest_lock:
        hit = _rest_cache.get(key)
        if hit and now - hit[0] < TTL:
            return hit[1]
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    with _rest_lock:
        _rest_cache[key] = (now, data)
        if len(_rest_cache) > 400:
            for k in sorted(_rest_cache, key=lambda k: _rest_cache[k][0])[:100]:
                _rest_cache.pop(k, None)
    return data


def _ms(t: pd.Timestamp) -> int:
    return int(t.timestamp() * 1000)


def _cached_parquet(path, cols: list):
    if not path.exists():
        return None
    key = (str(path), tuple(cols))
    mt = path.stat().st_mtime
    hit = _file_cache.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    df = pd.read_parquet(path, columns=cols)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    _file_cache[key] = (mt, df)
    return df


# ------------------------------------------------------------------ binance inputs
def _binance_klines_rest(symbol: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    rows, cur = [], _ms(t0)
    for _ in range(6):
        d = _get(FAPI + "/fapi/v1/klines", dict(symbol=symbol, interval="5m", startTime=cur,
                                                 endTime=_ms(t1), limit=1500))
        if not d:
            break
        rows += d
        cur = int(d[-1][0]) + 300_000
        if len(d) < 1500:
            break
    if not rows:
        return pd.DataFrame()
    a = np.array([[float(x[i]) for i in range(6)] for x in rows])
    return pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close", "volume"],
                        index=pd.to_datetime(a[:, 0], unit="ms", utc=True))


def _binance_oi_rest(symbol: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.Series:
    rows, cur = [], max(_ms(t0), _ms(pd.Timestamp.now("UTC") - pd.Timedelta("29D")))
    for _ in range(12):
        d = _get(FAPI + "/futures/data/openInterestHist",
                 dict(symbol=symbol, period="5m", limit=500, startTime=cur, endTime=_ms(t1)))
        if not d:
            break
        rows += d
        cur = int(d[-1]["timestamp"]) + 300_000
        if len(d) < 500:
            break
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([float(x["sumOpenInterest"]) for x in rows],
                     index=pd.to_datetime([int(x["timestamp"]) for x in rows], unit="ms", utc=True))


def _binance_5m(symbol: str, t0, t1, now, notes: list):
    """Local 5m klines + OI (archive, hist), tail filled from public REST. Returns (ohlcv, oi)."""
    base = loaders._base(symbol) / "um_futures"
    k = _cached_parquet(base / "klines_5m.parquet", ["open", "high", "low", "close", "volume"])
    k = pd.DataFrame() if k is None else k
    lim = min(t1, now)
    last = k.index.max() if len(k) else t0 - pd.Timedelta("5min")
    if last < lim - pd.Timedelta("10min"):
        try:
            tail = _binance_klines_rest(symbol, last + pd.Timedelta("5min"), t1)
            if len(tail):
                k = pd.concat([k[~k.index.isin(tail.index)], tail]).sort_index()
                notes.append("klines: tail via REST")
        except Exception as e:  # network optional
            notes.append(f"klines REST failed: {type(e).__name__}")
    k = k[(k.index >= t0) & (k.index + pd.Timedelta("5min") <= lim)]
    parts = []
    for fn, col in (("metrics_archive_5m.parquet", "sum_open_interest"),
                    ("openInterestHist_5m.parquet", "sumOpenInterest")):
        d = _cached_parquet(base / fn, [col])
        if d is not None and len(d):
            parts.append(d[col].rename("oi"))
    oi = parts[0] if parts else pd.Series(dtype=float, name="oi", index=pd.DatetimeIndex([], tz="UTC"))
    for p in parts[1:]:
        oi = oi.combine_first(p)
    last_oi = oi.index.max() if len(oi) else t0
    if last_oi < lim - pd.Timedelta("15min"):
        try:
            t = _binance_oi_rest(symbol, last_oi + pd.Timedelta("5min"), t1)
            if len(t):
                oi = oi.combine_first(t)
                notes.append("OI: tail via REST")
        except Exception as e:
            notes.append(f"OI REST failed: {type(e).__name__}")
    oi = oi[(oi.index >= t0 - pd.Timedelta("1D")) & (oi.index <= t1)].astype(float)
    return k, oi


def _to_res(k5: pd.DataFrame, oi5: pd.Series, res: str):
    if not len(k5):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turn"]), pd.Series(dtype=float)
    k5 = k5.assign(turn=k5["volume"] * k5["close"])   # taker-turnover proxy, USD per 5m bar
    bars = k5.resample(res, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
         "turn": "sum"}).dropna(subset=["close"])
    lim = max(2, int(pd.Timedelta("12h") / pd.Timedelta(res)))
    if len(oi5):
        o = oi5.resample(res, label="left", closed="left").last().ffill(limit=lim).reindex(bars.index)
    else:
        o = pd.Series(np.nan, index=bars.index)
    return bars, o


# ------------------------------------------------------------------ bybit inputs (optional venue)
def _bybit_pages(path: str, params: dict, t0, t1, limit: int):
    rows, end, n = [], _ms(t1), 0
    while end > _ms(t0) and n < 80:
        d = _get(BYBIT + path, {**params, "limit": limit, "endTime": end})
        lst = d["result"]["list"]
        if not lst:
            break
        rows += lst
        last = lst[-1]
        end = int(last[0] if isinstance(last, list) else last["timestamp"]) - 1
        n += 1
        if len(lst) < limit:
            break
    return rows


def _bybit(symbol: str, t0, t1, res: str, now, notes: list):
    kint = {"5min": "5", "15min": "15", "1h": "60"}[res]
    lim = min(t1, now)
    kr = _bybit_pages("/v5/market/kline", dict(category="linear", symbol=symbol, interval=kint), t0, lim, 1000)
    orow = _bybit_pages("/v5/market/open-interest",
                        dict(category="linear", symbol=symbol, intervalTime=res), t0, lim, 200)
    a = np.array([[float(x[i]) for i in range(6)] for x in kr]) if kr else np.zeros((0, 6))
    k = pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close", "volume"],
                     index=pd.to_datetime(a[:, 0], unit="ms", utc=True)).sort_index()
    k = k[~k.index.duplicated()]
    oi = pd.Series([float(x["openInterest"]) for x in orow],
                   index=pd.to_datetime([int(x["timestamp"]) for x in orow], unit="ms", utc=True)).sort_index()
    oi = oi[~oi.index.duplicated()]
    k = k[(k.index >= t0) & (k.index + pd.Timedelta(res) <= lim)]
    return k, oi


def _gate_oi(symbol: str) -> pd.Series:
    """Gate 5m OI converted to base-coin units (USD OI / mark) so the model's dOI x close holds."""
    from src.data import gate
    d = gate.load(symbol)
    if d is None or not len(d):
        raise FileNotFoundError(f"sem dados Gate para {symbol} (rode python -m src.data.gate)")
    return (d["open_interest_usd"] / d["mark_price"].where(d["mark_price"] > 0)).rename("oi")


def _okx_oi(symbol: str, t0, t1, pages: int = 12) -> pd.Series:
    """OKX swap OI (coin units, oiCcy) hourly from the public rubik endpoint (~100 h/page, capped)."""
    inst = symbol[:-4] + "-USDT-SWAP"
    rows, end = [], _ms(t1)
    for _ in range(pages):
        d = _get(OKX + "/api/v5/rubik/stat/contracts/open-interest-history",
                 dict(instId=inst, period="1H", end=end, limit=100))
        lst = d.get("data") or []
        if not lst:
            break
        rows += lst
        end = int(lst[-1][0]) - 1
        if end < _ms(t0) or len(lst) < 100:
            break
    if not rows:
        raise ValueError("OKX sem OI")
    s = pd.Series([float(x[2]) for x in rows], index=pd.to_datetime([int(x[0]) for x in rows], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index().rename("oi")


def venue_turnover(turn_b, oi_v, oi_b, dt_s: float) -> pd.Series:
    """Venue taker-turnover proxy = Binance turnover x (venue OI / Binance OI), ratio smoothed over ~1 day.
    Used for venues without their own volume feed (gate, okx); the fitted kappa is defined on this scale."""
    r = (oi_v / oi_b.where(oi_b > 0)).replace([np.inf, -np.inf], np.nan)
    w = max(1, int(86400 / dt_s))
    r = r.rolling(w, min_periods=1).mean().ffill().clip(0, 3).fillna(0.0)
    return turn_b * r


# ------------------------------------------------------------------ model
def liq_columns(close, high, low, oi, dt_s: float, edges: tuple, start: int,
                weights=None, mmr=MMR, long_share=0.5, turnover=None, kappa=0.0,
                half_life_h=None, scale=1.0):
    """Causal state machine over bars. Returns (grid float32 [T-start, P] long+short USD,
    final long, final short, events [(bar_idx, side, usd, price)], bin centers).
    edges=(ln lo, ln step, P): bin j covers ln p in [lo + j*step, lo + (j+1)*step).
    Gross-opening estimate (optional, defaults reproduce the legacy net-dOI model):
    new notional per bar = scale * (max(dOI,0)*close + kappa*turnover), turnover = taker USD volume per bar.
    half_life_h overrides HALF_LIFE_H."""
    weights = weights or DEFAULT_WEIGHTS
    close, high, low, oi = (np.asarray(a, float) for a in (close, high, low, oi))
    T = len(close)
    lo, dl, P = edges
    ws = np.array(list(weights.values()), float)
    ws /= ws.sum()
    levs = np.array(list(weights.keys()), float)
    centers = np.exp(lo + (np.arange(P) + 0.5) * dl)
    with np.errstate(invalid="ignore", divide="ignore"):
        li = np.floor((np.log(np.maximum(low, 1e-9)) - lo) / dl)
        hi = np.floor((np.log(np.maximum(high, 1e-9)) - lo) / dl)
        pl = close[:, None] * (1 - 1 / levs[None, :] + mmr)
        ps = close[:, None] * (1 + 1 / levs[None, :] - mmr)
        il = np.nan_to_num(np.floor((np.log(np.maximum(pl, 1e-9)) - lo) / dl), nan=-1).astype(np.int64)
        isr = np.nan_to_num(np.floor((np.log(np.maximum(ps, 1e-9)) - lo) / dl), nan=-1).astype(np.int64)
    li = np.nan_to_num(li, nan=0).astype(np.int64)
    hi = np.nan_to_num(hi, nan=-1).astype(np.int64)
    base_decay = 0.5 ** (dt_s / 3600.0 / (half_life_h or HALF_LIFE_H))
    to = None if turnover is None or not kappa else np.nan_to_num(np.asarray(turnover, float))
    L = np.zeros(P)
    S = np.zeros(P)
    grid = np.zeros((max(T - start, 0), P), np.float32)
    events = []
    prev = np.nan
    for t in range(T):
        g = base_decay
        if np.isfinite(oi[t]) and np.isfinite(prev) and oi[t] < prev and prev > 0:
            g *= oi[t] / prev
        L *= g
        S *= g
        a = max(int(li[t]), 0)          # longs at/above the bar low were traded through
        b = min(int(hi[t]) + 1, P)      # shorts at/below the bar high
        if a < P and L[a:].any():
            seg = L[a:]
            usd = float(seg.sum())
            if t >= start:
                events.append((t, "long", usd, float((seg * centers[a:]).sum() / usd)))
            L[a:] = 0.0
        if b > 0 and S[:b].any():
            seg = S[:b]
            usd = float(seg.sum())
            if t >= start:
                events.append((t, "short", usd, float((seg * centers[:b]).sum() / usd)))
            S[:b] = 0.0
        if np.isfinite(oi[t]):
            notional = 0.0
            if np.isfinite(prev) and np.isfinite(close[t]):
                notional = max(oi[t] - prev, 0.0) * close[t]
                if to is not None:
                    notional += kappa * to[t]
                notional *= scale
            if notional > 0:
                m = (il[t] >= 0) & (il[t] < P)
                np.add.at(L, il[t][m], notional * long_share * ws[m])
                m = (isr[t] >= 0) & (isr[t] < P)
                np.add.at(S, isr[t][m], notional * (1 - long_share) * ws[m])
            prev = oi[t]
        if t >= start:
            grid[t - start] = L + S
    return grid, L.copy(), S.copy(), events, centers


# ------------------------------------------------------------------ public API
def _encode(grid: np.ndarray):
    nz = grid[grid > 0]
    scale = max(float(np.quantile(nz, 0.998)) if nz.size else 1.0, 1.0)
    q = np.rint(255.0 * np.sqrt(np.minimum(grid / scale, 1.0))).astype(np.uint8)
    return q.tobytes(), scale


MODES = ("exposure", "liquidated", "raw")


def compute(symbol: str = "BTCUSDT", range_: str = "1w", end=None, venues=("binance",),
            calibrated: bool = True, mode: str | None = None) -> dict:
    """mode: "exposure" (default) = calibrated structure (gross openings, leverage mix, half-life)
    without the Gate scale, i.e. notional positioned at each liquidation price; "liquidated" = also
    scaled to the USD Gate actually liquidates; "raw" = original net-ΔOI model."""
    if mode is None:
        mode = "exposure" if calibrated else "raw"
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if range_ not in RANGES:
        raise ValueError(f"range must be one of {list(RANGES)}")
    venues = tuple(v for v in venues if v in VENUES) or ("binance",)
    win, res = RANGES[range_]
    now = pd.Timestamp.now("UTC")
    t1 = pd.Timestamp(end) if end is not None else now.ceil(res)
    t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
    wstart = t1 - pd.Timedelta(win)
    warm = max(2 * pd.Timedelta(win), MIN_WARM)
    t0 = (wstart - warm).floor(res)
    notes: list = []

    k5, oi5 = _binance_5m(symbol, t0, t1, now, notes)
    bars, oi_b = _to_res(k5, oi5, res)
    if len(bars) < 10:
        raise ValueError(f"sem dados para {symbol}")
    start = int(bars.index.searchsorted(wstart))
    window = bars.iloc[start:]
    if not len(window):
        raise ValueError("janela sem dados")
    lo_p, hi_p = float(window["low"].min()), float(window["high"].max())
    lr = np.log(hi_p / lo_p)
    pad = max(0.04, 0.15 * lr)
    edges = (np.log(lo_p) - pad, (lr + 2 * pad) / N_ROWS, N_ROWS)
    dt_s = pd.Timedelta(res).total_seconds()

    grid = None
    Lf = np.zeros(N_ROWS)
    Sf = np.zeros(N_ROWS)
    ev_all, used = [], []
    if mode == "raw":
        cal = dict(kappa=0.0, half_life_h=HALF_LIFE_H, scale=1.0, weights=DEFAULT_WEIGHTS)
    elif mode == "exposure":
        cal = {**CALIB, "scale": 1.0}
    else:
        cal = CALIB
    for v in venues:
        try:
            tn = None
            if v == "binance":
                b, o = bars, oi_b
                tn = bars["turn"]
            elif v in ("gate", "okx"):
                b = bars
                o = _to_res(k5, _gate_oi(symbol) if v == "gate" else _okx_oi(symbol, t0, t1), res)[1]
                tn = venue_turnover(bars["turn"], o, oi_b, dt_s)
            else:
                kb, ob = _bybit(symbol, t0, t1, res, now, notes)
                b, o = _to_res(kb, ob, res)
                b = b.reindex(bars.index)
                for c in ("open", "high", "low", "close"):
                    b[c] = b[c].fillna(bars[c])
                o = o.reindex(bars.index)
                tn = b["turn"].reindex(bars.index)
            g, L, S, ev, centers = liq_columns(
                b["close"], b["high"], b["low"], o, dt_s, edges, start, weights=cal["weights"],
                turnover=tn, kappa=cal["kappa"], half_life_h=cal["half_life_h"],
                scale=cal.get("scale_by_venue", {}).get(v, cal["scale"]))
        except Exception as e:
            if v == "binance":
                raise
            notes.append(f"{v} failed: {type(e).__name__}")
            continue
        grid = g if grid is None else grid + g
        Lf += L
        Sf += S
        ev_all += ev
        used.append(v)

    idx = window.index
    ev_all.sort(key=lambda e: -e[2])
    ev_all = sorted(ev_all[:MAX_EVENTS], key=lambda e: e[0])
    buf, scale = _encode(grid)
    active = [{"p": round(float(centers[j]), 2), "usd": round(float(x), 0), "side": s}
              for j in np.nonzero((Lf > 0) | (Sf > 0))[0]
              for x, s in ((Lf[j], "long"), (Sf[j], "short")) if x > 0]
    ts = (idx.asi8 // 10**9).tolist()
    return {
        "symbol": symbol, "range": range_, "resolution": res, "step": int(dt_s), "venues": used,
        "start": int(wstart.timestamp()), "end": int(t1.timestamp()),
        "times": ts, "prices": [round(float(c), 2) for c in centers],
        "shape": [len(idx), N_ROWS], "encoding": "u8 = round(255*sqrt(min(usd/scale,1)))", "scale": scale,
        "grid": base64.b64encode(buf).decode("ascii"),
        "candles": {k: window[c].round(2 if k != "v" else 3).tolist() for k, c in
                    (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume"))},
        "active": active,
        "events": [{"t": ts[e[0] - start], "p": round(e[3], 2), "usd": round(e[2], 0), "side": e[1]} for e in ev_all],
        "calibrated": mode != "raw",
        "mode": mode,
        "note": "Modelado a partir do delta de OI; consumido ao toque; não é previsão.",
        "notes": notes,
    }
