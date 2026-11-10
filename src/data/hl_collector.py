"""Hyperliquid exact-liquidation collector (public API, no key). Run: python -m src.data.hl_collector

(a) discovers trader addresses from the public `trades` websocket,
(b) polls `clearinghouseState` round-robin (priority: open positions / large notional),
(c) polls `frontendOpenOrders` for addresses with positions (stop / trigger orders),
(d) polls `metaAndAssetCtxs` for OI + mark.
Rate limit (docs): 1200 weight/min per IP; clearinghouseState=2, frontendOpenOrders=20, metaAndAssetCtxs=20.
We spend at most BUDGET_PER_MIN = 500 (<42%, under the 50% cap). Stored in SQLite (WAL, crash-safe).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "raw" / "hyperliquid" / "hl.sqlite"
COINS = ("BTC", "ETH", "SOL", "XRP", "BNB", "NEAR", "FET", "ADA")
WS_URL = "wss://api.hyperliquid.xyz/ws"
INFO_URL = "https://api.hyperliquid.xyz/info"
BUDGET_PER_MIN = 500.0
W_STATE, W_ORDERS, W_META = 2, 20, 20
ORDERS_SHARE = 0.40            # max share of the weight budget spent on open-order polls
BIG_NOTIONAL = 1_000_000.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS addresses(addr TEXT PRIMARY KEY, first_seen REAL, last_seen REAL,
  last_polled REAL DEFAULT 0, last_orders REAL DEFAULT 0, notional REAL DEFAULT 0, npos INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS positions(addr TEXT, coin TEXT, side TEXT, size REAL, entry_px REAL, leverage REAL,
  liq_px REAL, notional REAL, margin REAL, updated_at REAL, PRIMARY KEY(addr, coin));
CREATE INDEX IF NOT EXISTS pos_coin ON positions(coin);
CREATE TABLE IF NOT EXISTS orders(addr TEXT, oid INTEGER, coin TEXT, side TEXT, trigger_px REAL, size REAL,
  reduce_only INTEGER, order_type TEXT, updated_at REAL, PRIMARY KEY(addr, oid));
CREATE INDEX IF NOT EXISTS ord_coin ON orders(coin);
CREATE TABLE IF NOT EXISTS assetctx(coin TEXT PRIMARY KEY, oi REAL, mark REAL, oi_usd REAL, updated_at REAL);
"""
log = logging.getLogger("hl")


# ----------------------------------------------------------------- pure parsing
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_trades(msg: dict) -> set[str]:
    """Addresses (both counterparties) from a ws `trades` message."""
    if msg.get("channel") != "trades":
        return set()
    out = set()
    for t in msg.get("data") or []:
        for u in t.get("users") or []:
            if isinstance(u, str) and u.startswith("0x") and len(u) == 42:
                out.add(u.lower())
    return out


def parse_state(resp: dict, coins=COINS, now: float | None = None) -> list[dict]:
    """clearinghouseState -> list of position rows for tracked coins."""
    now = time.time() if now is None else now
    rows = []
    for ap in resp.get("assetPositions") or []:
        p = ap.get("position") or {}
        coin, szi = p.get("coin"), _f(p.get("szi"))
        if coin not in coins or not szi:
            continue
        lev = p.get("leverage") or {}
        rows.append(dict(coin=coin, side="long" if szi > 0 else "short", size=abs(szi), entry_px=_f(p.get("entryPx")),
                         leverage=_f(lev.get("value")), liq_px=_f(p.get("liquidationPx")),
                         notional=abs(_f(p.get("positionValue")) or 0.0), margin=_f(p.get("marginUsed")), updated_at=now))
    return rows


def parse_orders(resp: list, coins=COINS, now: float | None = None) -> list[dict]:
    """frontendOpenOrders -> trigger (stop / take-profit) orders for tracked coins."""
    now = time.time() if now is None else now
    rows = []
    for o in resp or []:
        if not o.get("isTrigger") or o.get("coin") not in coins:
            continue
        px, sz = _f(o.get("triggerPx")), _f(o.get("sz"))
        if not px:
            continue
        if not sz:  # position-tpsl orders report sz=0 ("whole position"): fall back to origSz if any
            sz = _f(o.get("origSz")) or 0.0
        rows.append(dict(oid=int(o["oid"]), coin=o["coin"], side="buy" if o.get("side") == "B" else "sell",
                         trigger_px=px, size=sz, reduce_only=int(bool(o.get("reduceOnly") or o.get("isPositionTpsl"))),
                         order_type=o.get("orderType") or "", updated_at=now))
    return rows


def parse_ctxs(resp, coins=COINS, now: float | None = None) -> list[dict]:
    """metaAndAssetCtxs -> [(coin, oi contracts, mark, oi usd)] for tracked coins."""
    now = time.time() if now is None else now
    meta, ctxs = resp
    out = []
    for u, c in zip(meta["universe"], ctxs):
        if u["name"] in coins:
            oi, mk = _f(c.get("openInterest")) or 0.0, _f(c.get("markPx")) or 0.0
            out.append(dict(coin=u["name"], oi=oi, mark=mk, oi_usd=oi * mk, updated_at=now))
    return out


# ----------------------------------------------------------------- storage
def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    return db


def add_addresses(db, addrs, now: float | None = None):
    now = time.time() if now is None else now
    db.executemany("INSERT INTO addresses(addr, first_seen, last_seen) VALUES(?,?,?) "
                   "ON CONFLICT(addr) DO UPDATE SET last_seen=excluded.last_seen", [(a, now, now) for a in addrs])
    db.commit()


def save_state(db, addr: str, rows: list[dict], now: float | None = None):
    now = time.time() if now is None else now
    db.execute("DELETE FROM positions WHERE addr=?", (addr,))
    db.executemany("INSERT INTO positions VALUES(?,?,?,?,?,?,?,?,?,?)",
                   [(addr, r["coin"], r["side"], r["size"], r["entry_px"], r["leverage"], r["liq_px"],
                     r["notional"], r["margin"], r["updated_at"]) for r in rows])
    db.execute("UPDATE addresses SET last_polled=?, notional=?, npos=? WHERE addr=?",
               (now, sum(r["notional"] for r in rows), len(rows), addr))
    db.commit()


def save_orders(db, addr: str, rows: list[dict], now: float | None = None):
    now = time.time() if now is None else now
    db.execute("DELETE FROM orders WHERE addr=?", (addr,))
    db.executemany("INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?)",
                   [(addr, r["oid"], r["coin"], r["side"], r["trigger_px"], r["size"], r["reduce_only"],
                     r["order_type"], r["updated_at"]) for r in rows])
    db.execute("UPDATE addresses SET last_orders=? WHERE addr=?", (now, addr))
    db.commit()


def save_ctxs(db, rows):
    db.executemany("INSERT OR REPLACE INTO assetctx VALUES(?,?,?,?,?)",
                   [(r["coin"], r["oi"], r["mark"], r["oi_usd"], r["updated_at"]) for r in rows])
    db.commit()


def next_state_addr(db):
    """Most overdue address: big holders every 60s, others with positions 5min, flat wallets 30min."""
    return db.execute(
        "SELECT addr, (CASE WHEN last_polled=0 THEN 0 WHEN notional>=? THEN last_polled+60 "
        "WHEN npos>0 THEN last_polled+300 ELSE last_polled+1800 END) AS due FROM addresses "
        "ORDER BY due LIMIT 1", (BIG_NOTIONAL,)).fetchone()


def next_orders_addr(db):
    """Addresses holding tracked positions: big every 5min, others every 20min; never-polled first."""
    return db.execute(
        "SELECT addr, (CASE WHEN last_orders=0 THEN 0 WHEN notional>=? THEN last_orders+300 ELSE last_orders+1200 END) "
        "AS due FROM addresses WHERE npos>0 ORDER BY due LIMIT 1", (BIG_NOTIONAL,)).fetchone()


# ----------------------------------------------------------------- runtime
class Limiter:
    """Serial weight pacer: never exceeds `per_min` weight per minute on average."""

    def __init__(self, per_min=BUDGET_PER_MIN):
        self.rate = per_min / 60.0
        self.free_at = 0.0

    async def spend(self, w: float):
        now = time.monotonic()
        wait = self.free_at - now
        if wait > 0:
            await asyncio.sleep(wait)
        self.free_at = max(now, self.free_at) + w / self.rate


class Collector:
    def __init__(self, db_path: Path = DB_PATH):
        self.db = connect(db_path)
        self.lim = Limiter()
        self.http: httpx.AsyncClient | None = None
        self.new: set[str] = set()
        self.spent = {"state": 0.0, "orders": 0.0}
        self.coins = COINS

    async def info(self, body: dict, weight: float):
        await self.lim.spend(weight)
        for attempt in range(4):
            try:
                r = await self.http.post(INFO_URL, json=body)
                if r.status_code == 429:
                    log.warning("429, backing off")
                    await asyncio.sleep(30 * (attempt + 1))
                    self.lim.free_at = time.monotonic() + 30
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError) as e:
                log.warning("info %s failed: %r", body.get("type"), e)
                await asyncio.sleep(5 * (attempt + 1))
        return None

    async def discover_coins(self):
        r = await self.info({"type": "meta"}, 20)
        if r:
            listed = {u["name"] for u in r["universe"]}
            self.coins = tuple(c for c in COINS if c in listed)
            log.info("coins on HL: %s", self.coins)

    async def ws_loop(self):
        from websockets.asyncio.client import connect as ws_connect
        backoff = 2
        while True:
            try:
                async with ws_connect(WS_URL, ping_interval=20, max_size=2**22) as ws:
                    for c in self.coins:
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": c}}))
                    log.info("ws connected, %d subs", len(self.coins))
                    backoff = 2
                    async for raw in ws:
                        self.new |= parse_trades(json.loads(raw))
            except Exception as e:  # noqa: BLE001 - reconnect on anything
                log.warning("ws error %r; reconnect in %ss", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)

    async def flush_loop(self):
        while True:
            await asyncio.sleep(5)
            if self.new:
                batch, self.new = self.new, set()
                add_addresses(self.db, batch)

    async def meta_loop(self):
        while True:
            r = await self.info({"type": "metaAndAssetCtxs"}, W_META)
            if r:
                save_ctxs(self.db, parse_ctxs(r, self.coins))
            await asyncio.sleep(300)

    async def _poll_orders(self, addr: str):
        r = await self.info({"type": "frontendOpenOrders", "user": addr}, W_ORDERS)
        self.spent["orders"] += W_ORDERS
        if r is not None:
            save_orders(self.db, addr, parse_orders(r, self.coins))

    async def poll_loop(self):
        while True:
            now = time.time()
            tot = self.spent["state"] + self.spent["orders"]
            if tot > 0 and self.spent["orders"] < ORDERS_SHARE * tot:
                row = next_orders_addr(self.db)
                if row and row[1] <= now:
                    await self._poll_orders(row[0])
                    continue
            row = next_state_addr(self.db)
            if row is None or row[1] > now:
                orow = next_orders_addr(self.db)
                if orow and orow[1] <= now:
                    await self._poll_orders(orow[0])
                else:
                    await asyncio.sleep(2)
                continue
            r = await self.info({"type": "clearinghouseState", "user": row[0]}, W_STATE)
            self.spent["state"] += W_STATE
            if r is not None:
                save_state(self.db, row[0], parse_state(r, self.coins))
            else:  # push it back so a bad address does not block the queue
                self.db.execute("UPDATE addresses SET last_polled=? WHERE addr=?", (time.time(), row[0]))
                self.db.commit()
            for k in self.spent:  # decay so the share adapts
                self.spent[k] *= 0.9995

    async def stats_loop(self):
        while True:
            await asyncio.sleep(120)
            a, p, n = (self.db.execute(q).fetchone()[0] for q in (
                "SELECT COUNT(*) FROM addresses", "SELECT COUNT(*) FROM positions", "SELECT COUNT(*) FROM orders"))
            log.info("addresses=%d positions=%d trigger_orders=%d", a, p, n)

    async def run(self):
        async with httpx.AsyncClient(timeout=20) as self.http:
            await self.discover_coins()
            await asyncio.gather(self.ws_loop(), self.flush_loop(), self.meta_loop(), self.poll_loop(), self.stats_loop())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    while True:  # crash-safe: restart the whole service on unexpected errors
        try:
            asyncio.run(Collector().run())
        except KeyboardInterrupt:
            return
        except Exception:  # noqa: BLE001
            log.exception("collector crashed; restarting in 10s")
            time.sleep(10)


if __name__ == "__main__":
    main()
