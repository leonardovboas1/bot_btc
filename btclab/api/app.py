"""FastAPI app. Backtest stats use in-sample data only (bars < holdout cutoff)."""
from __future__ import annotations

import inspect
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
def _full(market: str, tf: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """All bars incl. holdout (treat as read-only); perp gets a funding column."""
    try:
        df = loaders.load_ohlcv(market, tf, allow_holdout=True, symbol=symbol)
        if market == "perp":
            df = df.copy()
            df["funding"] = loaders.load_funding(tf, allow_holdout=True, symbol=symbol).values
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return df


def _sym(symbol: str) -> str:
    return symbol.upper()


def _liq_kw(fn, symbol: str) -> dict:
    """Pass symbol to a liq_service function only if it accepts it."""
    return {"symbol": symbol} if "symbol" in inspect.signature(fn).parameters else {}


def _liq_ok(fn, symbol: str) -> bool:
    return symbol == "BTCUSDT" or "symbol" in inspect.signature(fn).parameters


def _empty_liq(kind: str) -> dict:
    if kind == "book":
        return {"empty": True, "times": [], "prices": [], "bid": [], "ask": []}
    if kind == "liq":
        return {"empty": True, "times": [], "prices": [], "long": [], "short": [], "flush_times": []}
    return {"empty": True, "prices": [], "volume": []}


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
@app.get("/symbols")
def symbols():
    out = []
    for sym in loaders.list_symbols():
        base = loaders._base(sym) / "um_futures"
        tfs = [tf for tf in ("5m", "15m", "1h", "4h", "1d") if (base / f"klines_{tf}.parquet").exists()]
        if "1d" in tfs:
            tfs.append("1w")
        out.append({"symbol": sym, "base": sym[:-4] if sym.endswith("USDT") else sym, "tfs": tfs,
                    "has_liq": (base / "metrics_archive_5m.parquet").exists()})
    return out


@app.get("/ohlcv")
def ohlcv(market: str = "perp", tf: str = "4h", start: Optional[str] = None,
          end: Optional[str] = None, symbol: str = "BTCUSDT", limit: int = 5000):
    """Bars in [start, end). Without start, the last ``limit`` bars (limit=0: all). Bars after the
    last on-disk bar come from Binance REST (display only)."""
    symbol = _sym(symbol)
    df = _full(market, tf, symbol)[["open", "high", "low", "close", "volume"]]
    if len(df):
        last = df.index[-1]
        if end is None or _local(df, pd.Timestamp(end, tz="UTC")) > last:
            from btclab.api import live
            extra = live.tail(market, tf, symbol, last)
            if len(extra):
                if df.index.tz is None:
                    extra = extra.tz_convert("UTC").tz_localize(None)
                df = pd.concat([df, extra])
    if start is not None:
        df = df[df.index >= _local(df, pd.Timestamp(start, tz="UTC"))]
    if end is not None:
        df = df[df.index < _local(df, pd.Timestamp(end, tz="UTC"))]
    if start is None and limit and limit > 0:
        df = df.iloc[-limit:]
    t = _ts(df.index)
    cols = [df[c].to_numpy(dtype=float).tolist() for c in ("open", "high", "low", "close", "volume")]
    bars = [{"time": int(a), "open": o, "high": h, "low": l, "close": c, "volume": v}
            for a, o, h, l, c, v in zip(t.tolist(), *cols)]
    return {"symbol": symbol, "market": market, "tf": tf, "holdout_start": int(_cutoff().timestamp()), "bars": bars}


@app.get("/strategies")
def strategies():
    return [get_preset(k) for k in PRESETS]


class BacktestReq(BaseModel):
    strategy: Optional[str] = None
    spec: Optional[dict] = None
    market: str = "perp"
    symbol: str = "BTCUSDT"
    tf: Optional[str] = None
    fee: Optional[str] = None      # cost preset name, e.g. "binance_perp_taker"
    source: Optional[str] = None   # signal candles: "real" (default) or "ha" (Heikin Ashi); fills always real


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


MAX_CURVE_POINTS = 3000


def _daily(index, eq, bh) -> list:
    """Exact last value of each UTC day (for periodic P&L, Sharpe and benchmarking)."""
    d = pd.DataFrame({"e": np.asarray(eq, float), "b": np.asarray(bh, float)}, index=pd.DatetimeIndex(index))
    d = d.groupby(d.index.normalize()).last()
    return [{"time": int(a), "equity": float(e), "buy_hold": float(b)}
            for a, e, b in zip(_ts(d.index).tolist(), d["e"], d["b"])]


def _curve_idx(v: np.ndarray, max_points: int = MAX_CURVE_POINTS) -> np.ndarray:
    """Indices that keep the shape of a long curve: first, last, and each bucket's min and max."""
    n = len(v)
    if n <= max_points:
        return np.arange(n)
    edges = np.linspace(0, n, max_points // 2 + 1).astype(int)
    keep = {0, n - 1}
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            seg = v[a:b]
            keep.add(a + int(np.argmin(seg)))
            keep.add(a + int(np.argmax(seg)))
    return np.array(sorted(keep))


@app.post("/backtest")
def backtest(req: BacktestReq):
    spec = _resolve_spec(req.strategy, req.spec, req.tf)
    if req.fee:
        spec["costs"] = req.fee
    if req.source:
        spec["source"] = req.source
    key = json.dumps([spec, _sym(req.symbol), req.market], sort_keys=True, default=str)
    return _backtest_cached(key)


@lru_cache(maxsize=64)
def _backtest_cached(key: str):
    spec, symbol, market = json.loads(key)
    tf = spec["tf"]
    full = _full(market, tf, symbol)
    n_is = int(full.index.searchsorted(_local(full, _cutoff())))
    if n_is < 50:
        raise HTTPException(400, "not enough in-sample data")
    ins = full.iloc[:n_is]
    res, metrics = _run(spec, ins)
    t = _ts(ins.index)
    eq = res.equity
    close = ins["close"].to_numpy(dtype=float)
    bh = eq[0] * close / close[0]
    keep = np.union1d(_curve_idx(np.asarray(eq, float), MAX_CURVE_POINTS // 2),
                      _curve_idx(np.asarray(bh, float), MAX_CURVE_POINTS // 2))

    # Holdout: signals only (run over the full series so state is continuous); no stats.
    hold_markers: list = []
    if len(full) > n_is:
        fres, _ = _run(spec, full)
        hold_markers = _markers(full, fres.trades[fres.trades["entry_idx"] >= n_is])
    return _num({
        "strategy": spec["name"], "symbol": symbol, "market": market, "tf": tf,
        "holdout_start": int(_cutoff().timestamp()),
        "trades": _trade_rows(ins, res.trades),
        "markers": _markers(ins, res.trades),
        "equity": [{"time": int(t[i]), "value": float(eq[i])} for i in keep],
        "buy_hold": [{"time": int(t[i]), "value": float(bh[i])} for i in keep],
        "equity_daily": _daily(ins.index, eq, bh),
        "metrics": metrics,
        "holdout_markers": hold_markers,
    })


@app.get("/signal/current")
def signal_current(strategy: str = "brayvb_v2", tf: Optional[str] = None, market: str = "perp",
                   symbol: str = "BTCUSDT", source: Optional[str] = None):
    spec = _resolve_spec(strategy, None, tf)
    if source:
        spec["source"] = source
    df = _full(market, spec["tf"], _sym(symbol))
    res, _ = _run(spec, df)
    pos = int(res.position[-1])
    return {"strategy": strategy, "tf": spec["tf"],
            "state": {1: "LONG", -1: "SHORT"}.get(pos, "FLAT"),
            "time": int(_ts(df.index[-1:])[0]), "price": float(df["close"].iloc[-1])}


def _with_tail(market: str, tf: str, symbol: str) -> pd.DataFrame:
    """Cached on-disk bars plus live bars after them (display only)."""
    df = _full(market, tf, symbol)
    from btclab.api import live
    extra = live.tail(market, tf, symbol, df.index[-1]) if len(df) else None
    if extra is not None and len(extra):
        if df.index.tz is None:
            extra = extra.tz_convert("UTC").tz_localize(None)
        df = pd.concat([df, extra])
    return df


_IND_CACHE: dict = {}
IND_WARMUP = 4000


@app.get("/indicator/{name}")
def indicator(name: str, params: str = Query("{}", description="JSON object"),
              tf: str = "4h", market: str = "perp", symbol: str = "BTCUSDT",
              start: Optional[str] = None, limit: int = 5000):
    """Indicator series computed on the full history (+ live tail) so warm-up is exact; returns
    bars from ``start``, else the last ``limit`` bars (limit=0: all)."""
    if name not in REGISTRY:
        raise HTTPException(404, f"unknown indicator {name!r}")
    try:
        p = json.loads(params)
        if not isinstance(p, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "params must be a JSON object")
    symbol = _sym(symbol)
    df = _with_tail(market, tf, symbol)
    if start is None and limit and limit > 0:
        df = df.iloc[-(limit + IND_WARMUP):]   # display: a long warm-up makes recursive filters converge
    key = (name, json.dumps(p, sort_keys=True), tf, market, symbol, start, limit)
    hit = _IND_CACHE.get(key)
    if hit and hit[0] == (len(df), df.index[-1], float(df["close"].iloc[-1])):
        out = hit[1]
    else:
        try:
            out = compute(name, df, **p)
        except TypeError as e:
            raise HTTPException(400, str(e))
        if len(_IND_CACHE) > 64:
            _IND_CACHE.clear()
        _IND_CACHE[key] = ((len(df), df.index[-1], float(df["close"].iloc[-1])), out)
    t = _ts(df.index)
    lo = 0
    if start is not None:
        lo = int(df.index.searchsorted(_local(df, pd.Timestamp(start, tz="UTC"))))
    elif limit and limit > 0:
        lo = max(0, len(t) - limit)
    series = {}
    for k, v in out.items():
        a = np.asarray(v)
        if a.ndim != 1 or len(a) != len(t) or a.dtype == bool:
            continue
        a = a[lo:].astype(float)
        tt = t[lo:]
        ok = np.isfinite(a)
        series[k] = [{"time": int(x), "value": float(y)} for x, y in zip(tt[ok].tolist(), a[ok].tolist())]
    return {"name": name, "tf": tf, "symbol": symbol, "series": series}


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
                 rows: int = MAX_ROWS, cols: int = MAX_COLS, symbol: str = "BTCUSDT"):
    symbol = _sym(symbol)
    s, e = _window(tf, start, end)
    if not _liq_ok(liq_service.depth_heatmap_payload, symbol):
        return _empty_liq("book")
    lo, hi = _book_range()
    avail = {"available_from": lo, "available_to": hi}
    if int(pd.Timestamp(e, tz="UTC").timestamp()) < lo or int(pd.Timestamp(s, tz="UTC").timestamp()) > hi:
        return {"empty": True, **avail, "times": [], "prices": [], "bid": [], "ask": []}
    try:
        d = liq_service.depth_heatmap_payload(s, e, tf, **_liq_kw(liq_service.depth_heatmap_payload, symbol))
    except (FileNotFoundError, ValueError, KeyError) as ex:
        raise HTTPException(400, str(ex))
    if not d["times"]:
        return {"empty": True, **avail, "times": [], "prices": [], "bid": [], "ask": []}
    cols, rows = _limit(cols, rows)
    return {"empty": False, **avail, "unit": "USD milhões",
            **_shrink(d["times"], d["prices"], {"bid": d["bid"], "ask": d["ask"]}, cols, rows)}


@app.get("/heatmap/liq")
def heatmap_liq(symbol: str = "BTCUSDT", range: str = "1w", end: Optional[str] = None,
                venues: str = "binance", calibrated: bool = True, mode: Optional[str] = None):
    """Coinglass-style liquidation map (modelo OI bruto, consumido ao toque; calibrated=true aplica o ajuste Gate). Ver btclab/liquidity/liq_map.py."""
    from btclab.liquidity import liq_map
    symbol = _sym(symbol)
    if range not in liq_map.RANGES:
        raise HTTPException(400, f"range deve ser um de {list(liq_map.RANGES)}")
    vs = tuple(v for v in venues.split(",") if v in liq_map.VENUES) or ("binance",)
    try:
        end_ts = pd.Timestamp(end) if end else None
        res = liq_map.RANGES[range][1]
        if end_ts is not None:
            end_ts = (end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")).floor("h")
            bucket = int(end_ts.timestamp())
        else:
            bucket = int(pd.Timestamp.now("UTC").floor("5min").timestamp())  # live: refresh every 5 min
        mode = mode or ("exposure" if calibrated else "raw")
        return _liq_cached(symbol, range, bucket if end_ts is not None else None, bucket, vs, end_ts, res, mode)
    except (FileNotFoundError, ValueError, KeyError) as ex:
        raise HTTPException(400, str(ex))


_LIQ_CACHE: dict = {}


def _liq_cached(symbol, range_, end_key, bucket, vs, end_ts, res, mode="exposure"):
    key = (symbol, range_, bucket, vs, mode)
    hit = _LIQ_CACHE.get(key)
    if hit is None:
        from btclab.liquidity import liq_map
        hit = liq_map.compute(symbol, range_, end_ts, vs, mode=mode)
        for k in [k for k in _LIQ_CACHE if len(_LIQ_CACHE) > 24 and k != key][:8]:
            _LIQ_CACHE.pop(k, None)
        _LIQ_CACHE[key] = hit
    return hit


@app.get("/volume_profile")
def volume_profile(tf: str = "1h", start: Optional[str] = None, end: Optional[str] = None,
                   rows: int = MAX_ROWS, symbol: str = "BTCUSDT"):
    symbol = _sym(symbol)
    s, e = _window(tf, start, end)
    if not _liq_ok(liq_service.volume_profile_payload, symbol):
        return _empty_liq("vp")
    try:
        d = liq_service.volume_profile_payload(s, e, tf, **_liq_kw(liq_service.volume_profile_payload, symbol))
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


@app.get("/liquidations")
def liquidations(symbol: str = "BTCUSDT", start: Optional[str] = None, end: Optional[str] = None,
                 venues: str = "binance,bybit,okx,gate"):
    """Liquidacoes observadas: eventos (recorder binance/bybit/okx) + barras 5m (Gate). Default: ultimas 24h."""
    from btclab.liquidity import observed
    symbol = _sym(symbol)
    if not symbol.isalnum():
        raise HTTPException(400, "symbol invalido")
    try:
        t1 = pd.Timestamp(end) if end else pd.Timestamp.now("UTC")
        t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
        t0 = pd.Timestamp(start) if start else t1 - pd.Timedelta("1D")
        vs = tuple(v for v in venues.split(",") if v)
        return observed.observed(symbol, t0, t1, vs)
    except (ValueError, TypeError) as ex:
        raise HTTPException(400, str(ex))


def _hl_coin(coin: str):
    from btclab.liquidity import hl_levels
    c = hl_levels.coin_of(coin)
    if c not in hl_levels.COINS:
        raise HTTPException(400, f"coin deve ser um de {list(hl_levels.COINS)}")
    return hl_levels, c


@app.get("/hl/liqlevels")
def hl_liqlevels(coin: str = "BTC", bins: int = 240):
    """Niveis de liquidacao EXATOS (Hyperliquid, on-chain): notional por faixa de preco (long/short) + cobertura."""
    m, c = _hl_coin(coin)
    return _num(m.liq_levels(c, bins))


@app.get("/hl/stops")
def hl_stops(coin: str = "BTC", bins: int = 240):
    """Ordens trigger (stop/TP) abertas na Hyperliquid, agregadas por preco."""
    m, c = _hl_coin(coin)
    return _num(m.stops(c, bins))


# ------------------------------------------------------------------ forward paper trading
_PAPER_REMOTE = "https://raw.githubusercontent.com/leonardovboas1/bot_btc/paper-data/paper.sqlite"
_paper_pull = {"t": 0.0}


def _paper_sync(p):
    """When the ledger is hosted by the GitHub workflow (marker data/paper/REMOTE), refresh the
    local copy at most every 5 min. Failures keep the last copy."""
    import time as _t
    if not (p.parent / "REMOTE").exists() or _t.time() - _paper_pull["t"] < 300:
        return
    _paper_pull["t"] = _t.time()
    try:
        import requests
        r = requests.get(_PAPER_REMOTE, timeout=20)
        r.raise_for_status()
        if r.content[:16] != b"SQLite format 3\x00":
            return
        tmp = p.with_suffix(".new")
        tmp.write_bytes(r.content)
        for x in (p.with_name(p.name + "-wal"), p.with_name(p.name + "-shm")):
            x.unlink(missing_ok=True)
        tmp.replace(p)
    except Exception:  # noqa: BLE001
        pass


def _paper_db():
    import sqlite3
    from pathlib import Path
    p = Path(loaders.ROOT) / "data" / "paper" / "paper.sqlite"
    _paper_sync(p)
    if not p.exists():
        return None
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _paper_rows(con, sql, args=()):
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def _max_dd(xs):
    peak, dd = None, 0.0
    for x in xs:
        if x is None or x <= 0:
            continue
        peak = x if peak is None else max(peak, x)
        dd = max(dd, 1 - x / peak)
    return dd


def _paper_wins(con, sid):
    """(winning, closed) round trips from the fills ledger, net of fees."""
    rows = con.execute("SELECT trade_no,leg,position_side,qty,price,fee FROM fills WHERE strategy=? "
                       "ORDER BY trade_no, rowid", (sid,)).fetchall()
    trades: dict = {}
    for no, leg, side, qty, px, fee in rows:
        t = trades.setdefault(no, {"side": side, "in": None, "out": None, "fee": 0.0, "qty": qty})
        t["fee"] += fee or 0.0
        t["in" if leg == "entry" else "out"] = px
    closed = [t for t in trades.values() if t["in"] is not None and t["out"] is not None]
    sgn = lambda t: 1 if str(t["side"]).lower() in ("long", "1", "buy") else -1
    wins = sum(1 for t in closed if sgn(t) * (t["out"] - t["in"]) * abs(t["qty"] or 0) - t["fee"] > 0)
    return wins, len(closed)


@app.get("/paper/strategies")
def paper_strategies():
    con = _paper_db()
    if con is None:
        return []
    try:
        out = []
        for s in _paper_rows(con, "SELECT id,symbol,tf,source,spec,start_ts FROM strategies ORDER BY rowid"):
            sid = s["id"]
            eq = _paper_rows(con, "SELECT bar_time,equity,price FROM equity WHERE strategy=? ORDER BY bar_time",
                             (sid,))
            pos = con.execute("SELECT position,qty,entry_px,unrealized,bar_time FROM positions WHERE strategy=? "
                              "ORDER BY bar_time DESC LIMIT 1", (sid,)).fetchone()
            sig = con.execute("SELECT bar_time,bar_close_time,long_sig,short_sig,exit_long,exit_short FROM signals "
                              "WHERE strategy=? ORDER BY bar_time DESC LIMIT 1", (sid,)).fetchone()
            n_tr = con.execute("SELECT COUNT(*) FROM fills WHERE strategy=? AND leg='exit'", (sid,)).fetchone()[0]
            ent = con.execute("SELECT COUNT(*) FROM fills WHERE strategy=? AND leg='entry'", (sid,)).fetchone()[0]
            e0, e1 = (eq[0], eq[-1]) if eq else (None, None)
            ret = bh = mdd = bh_mdd = None
            if e0:
                ret = e1["equity"] / e0["equity"] - 1
                bh = e1["price"] / e0["price"] - 1
                mdd, bh_mdd = _max_dd([r["equity"] for r in eq]), _max_dd([r["price"] for r in eq])
            wins, closed = _paper_wins(con, sid)
            spec = json.loads(s["spec"])
            out.append({
                "id": sid, "name": spec.get("name"), "symbol": s["symbol"], "tf": s["tf"],
                "source": s["source"], "start_ts": s["start_ts"],
                "first_bar": e0["bar_time"] if e0 else None,
                "position": {1: "LONG", -1: "SHORT"}.get(pos["position"] if pos else 0, "FLAT"),
                "qty": pos["qty"] if pos else 0.0, "entry_px": pos["entry_px"] if pos else 0.0,
                "unrealized": pos["unrealized"] if pos else 0.0,
                "n_trades": n_tr, "n_entries": ent,
                "equity": e1["equity"] if e1 else None, "equity_start": e0["equity"] if e0 else None,
                "price": e1["price"] if e1 else None,
                "return": ret, "bh_return": bh, "mdd": mdd, "bh_mdd": bh_mdd,
                "wins": wins, "closed": closed,
                "days": ((e1["bar_time"] - e0["bar_time"]) / 86400.0) if e0 else 0.0,
                "last_signal_time": sig["bar_time"] if sig else None,
                "last_signal_close_time": sig["bar_close_time"] if sig else None,
                "last_signal": ({"long": sig["long_sig"], "short": sig["short_sig"],
                                 "exit_long": sig["exit_long"], "exit_short": sig["exit_short"]} if sig else None),
            })
        return _num(out)
    finally:
        con.close()


def _paper_known(con, strategy: str):
    if not con.execute("SELECT 1 FROM strategies WHERE id=?", (strategy,)).fetchone():
        raise HTTPException(404, f"unknown paper strategy {strategy!r}")


@app.get("/paper/ledger")
def paper_ledger(strategy: str, limit: int = 500):
    """Append-only ledger of one strategy: signals (last ``limit``), orders, fills, funding, positions
    (last ``limit``), plus chart markers for the fills and closed/open trades."""
    con = _paper_db()
    if con is None:
        raise HTTPException(404, "paper ledger not created yet")
    try:
        _paper_known(con, strategy)

        def q(t, extra=""):
            return _paper_rows(con, f"SELECT * FROM {t} WHERE strategy=? {extra}", (strategy,))

        lim = f"ORDER BY bar_time DESC LIMIT {int(limit)}"
        fills = q("fills", "ORDER BY bar_time, leg DESC")
        markers = []
        for f in fills:
            long_ = f["position_side"] == 1
            buy = f["side"] == "buy"
            markers.append({"time": f["bar_time"], "position": "belowBar" if buy else "aboveBar",
                            "color": GREEN if buy else RED, "shape": "arrowUp" if buy else "arrowDown",
                            "text": ("L" if long_ else "S") + (" in" if f["leg"] == "entry" else " out"),
                            "price": f["price"]})
        by = {}
        for f in fills:
            by.setdefault(f["trade_no"], {})[f["leg"]] = f
        trades = []
        for k, d in sorted(by.items()):
            e, x = d.get("entry"), d.get("exit")
            side = e["position_side"]
            r = {"trade_no": k, "side": "LONG" if side == 1 else "SHORT", "entry_time": e["bar_time"],
                 "entry_px": e["price"], "qty": e["qty"], "exit_time": x["bar_time"] if x else None,
                 "exit_px": x["price"] if x else None, "exit_reason": x["reason"] if x else None}
            if x:
                fund = con.execute("SELECT COALESCE(SUM(amount),0) FROM funding WHERE strategy=? AND bar_time>=? "
                                   "AND bar_time<=?", (strategy, e["bar_time"], x["bar_time"])).fetchone()[0]
                pnl = (x["price"] - e["price"]) * e["qty"] * side - e["fee"] - x["fee"] - fund
                r.update(pnl=pnl, pnl_pct=pnl / e["notional"])
            trades.append(r)
        return _num({
            "strategy": strategy,
            "signals": list(reversed(q("signals", lim))), "orders": q("orders", "ORDER BY fill_bar_time"),
            "fills": fills, "funding": q("funding", "ORDER BY bar_time"),
            "positions": list(reversed(q("positions", lim))), "trades": trades, "markers": markers})
    finally:
        con.close()


@app.get("/paper/equity")
def paper_equity(strategy: str):
    """Equity at every processed bar close, with buy & hold of the same symbol from the same first bar."""
    con = _paper_db()
    if con is None:
        raise HTTPException(404, "paper ledger not created yet")
    try:
        _paper_known(con, strategy)
        rows = _paper_rows(con, "SELECT bar_time,equity,price,position FROM equity WHERE strategy=? "
                                "ORDER BY bar_time", (strategy,))
        if not rows:
            return {"strategy": strategy, "points": [], "start": None}
        e0, p0 = rows[0]["equity"], rows[0]["price"]
        pts = [{"time": r["bar_time"], "equity": r["equity"], "buy_hold": e0 * r["price"] / p0,
                "price": r["price"], "position": r["position"]} for r in rows]
        return _num({"strategy": strategy, "start": rows[0]["bar_time"], "points": pts})
    finally:
        con.close()


@app.get("/paper/status")
def paper_status():
    """Last service run, lag (seconds since it) and recent errors."""
    import time as _t
    empty = {"running": False, "last_run": None, "lag_s": None, "errors": [], "last_bars": {}}
    con = _paper_db()
    if con is None:
        return empty
    try:
        r = con.execute("SELECT ts,started_at,duration_s,n_strategies,n_new_rows,errors,last_bars FROM runs "
                        "ORDER BY ts DESC LIMIT 1").fetchone()
        if not r:
            return empty
        errs = []
        for e in con.execute("SELECT ts,errors FROM runs WHERE errors != '[]' ORDER BY ts DESC LIMIT 20"):
            errs += [{"time": e["ts"], "error": m} for m in json.loads(e["errors"])]
        lag = _t.time() - r["ts"]
        return _num({"running": lag < (5400 if (__import__("pathlib").Path(loaders.ROOT) / "data" / "paper" / "REMOTE").exists() else 180), "last_run": r["ts"], "last_run_iso": r["started_at"],
                     "lag_s": lag, "duration_s": r["duration_s"], "n_strategies": r["n_strategies"],
                     "n_new_rows": r["n_new_rows"], "current_errors": json.loads(r["errors"]),
                     "errors": errs[:20], "last_bars": json.loads(r["last_bars"])})
    finally:
        con.close()
