"""Aggregation of the Hyperliquid collector DB (src/data/hl_collector.py) into exact liquidation / stop levels."""
from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from statistics import median

import numpy as np

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hyperliquid" / "hl.sqlite"
COINS = ("BTC", "ETH", "SOL", "XRP", "BNB", "NEAR", "FET", "ADA")
SPAN = 0.4   # price window: mark * (1 -/+ SPAN)


def coin_of(symbol: str) -> str:
    s = symbol.upper()
    for suf in ("USDT", "USDC", "USD", "PERP"):
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def open_db(path: Path = DB_PATH):
    if not Path(path).exists():
        return None
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    return db


def _edges(mark: float, bins: int) -> np.ndarray:
    return np.geomspace(mark * (1 - SPAN), mark * (1 + SPAN), bins + 1)


def aggregate(prices, usd, ages, edges):
    """Bin (price, usd, age) triples; returns per-bin (usd, n, median age)."""
    n = len(edges) - 1
    idx = np.searchsorted(edges, prices, side="right") - 1
    tot, cnt, ag = np.zeros(n), np.zeros(n, dtype=int), [[] for _ in range(n)]
    for i, p, u, a in zip(idx, prices, usd, ages):
        if 0 <= i < n and p < edges[-1]:
            tot[i] += u
            cnt[i] += 1
            ag[i].append(a)
    return tot, cnt, [float(median(a)) if a else None for a in ag]


def _ctx(db, coin):
    r = db.execute("SELECT oi, mark, oi_usd, updated_at FROM assetctx WHERE coin=?", (coin,)).fetchone()
    return None if r is None else dict(oi=r[0], mark=r[1], oi_usd=r[2], updated_at=r[3])


def liq_levels(coin: str, bins: int = 240, db=None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    db = db or open_db()
    if db is None:
        return {"coin": coin, "empty": True, "reason": "collector sem dados"}
    ctx = _ctx(db, coin)
    rows = db.execute("SELECT side, liq_px, notional, updated_at FROM positions WHERE coin=? AND liq_px>0", (coin,)).fetchall()
    n_addr = db.execute("SELECT COUNT(*) FROM addresses").fetchone()[0]
    n_polled = db.execute("SELECT COUNT(*) FROM addresses WHERE last_polled>0").fetchone()[0]
    total = sum(r[2] for r in db.execute("SELECT side, liq_px, notional FROM positions WHERE coin=?", (coin,)).fetchall())
    mark = (ctx or {}).get("mark") or (float(np.median([r[1] for r in rows])) if rows else 0.0)
    ages = [now - r[3] for r in rows]
    cov = dict(addresses=n_addr, polled=n_polled, positions=len(rows), notional=total,
               oi_usd=(ctx or {}).get("oi_usd"),
               oi_share=(total / ctx["oi_usd"] if ctx and ctx["oi_usd"] else None),
               median_age_s=float(median(ages)) if ages else None)
    if not rows or mark <= 0:
        return {"coin": coin, "empty": True, "mark": mark, "coverage": cov, "levels": []}
    bins = max(20, min(int(bins), 1000))
    edges = _edges(mark, bins)
    out = {}
    for side in ("long", "short"):
        sel = [r for r in rows if r[0] == side]
        out[side] = aggregate(np.array([r[1] for r in sel], float), [r[2] for r in sel], [now - r[3] for r in sel], edges)
    levels = []
    for i in range(bins):
        lu, su = out["long"][0][i], out["short"][0][i]
        if lu <= 0 and su <= 0:
            continue
        levels.append(dict(p=float(np.sqrt(edges[i] * edges[i + 1])), lo=float(edges[i]), hi=float(edges[i + 1]),
                           long_usd=float(lu), short_usd=float(su), n_long=int(out["long"][1][i]), n_short=int(out["short"][1][i]),
                           age_long_s=out["long"][2][i], age_short_s=out["short"][2][i]))
    return {"coin": coin, "empty": False, "mark": mark, "bins": bins, "levels": levels, "coverage": cov, "now": now}


def stops(coin: str, bins: int = 240, db=None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    db = db or open_db()
    if db is None:
        return {"coin": coin, "empty": True, "reason": "collector sem dados"}
    ctx = _ctx(db, coin)
    rows = db.execute("SELECT side, trigger_px, size, reduce_only, updated_at FROM orders WHERE coin=? AND trigger_px>0", (coin,)).fetchall()
    mark = (ctx or {}).get("mark") or (float(np.median([r[1] for r in rows])) if rows else 0.0)
    if not rows or mark <= 0:
        return {"coin": coin, "empty": True, "mark": mark, "levels": []}
    bins = max(20, min(int(bins), 1000))
    edges = _edges(mark, bins)
    res = {}
    for side in ("buy", "sell"):
        sel = [r for r in rows if r[0] == side]
        res[side] = aggregate(np.array([r[1] for r in sel], float), [r[1] * r[2] for r in sel], [now - r[4] for r in sel], edges)
    levels = []
    for i in range(bins):
        bu, se = res["buy"][0][i], res["sell"][0][i]
        if bu <= 0 and se <= 0:
            continue
        levels.append(dict(p=float(np.sqrt(edges[i] * edges[i + 1])), lo=float(edges[i]), hi=float(edges[i + 1]),
                           buy_usd=float(bu), sell_usd=float(se), n_buy=int(res["buy"][1][i]), n_sell=int(res["sell"][1][i]),
                           age_buy_s=res["buy"][2][i], age_sell_s=res["sell"][2][i]))
    return {"coin": coin, "empty": False, "mark": mark, "bins": bins, "levels": levels, "n_orders": len(rows), "now": now}
