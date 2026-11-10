"""Observed-liquidation recorder (Binance + Bybit websockets, OKX polling). Free public feeds, no keys.

Run (long-lived, from project root):  .venv/Scripts/python -m src.data.liq_recorder
Rows: ts (UTC ms), venue, symbol (e.g. BTCUSDT), side (side of the LIQUIDATED position: long|short),
price, qty (base coin), usd -> data/raw/liquidations/<venue>/<YYYY-MM-DD>.parquet.
Buffered, flushed every 60 s via atomic tmp+replace (a crash loses at most one buffer).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "liquidations"
BASES = ("BTC", "ETH", "SOL", "BNB", "NEAR", "FET", "ADA", "XRP")
SYMBOLS = tuple(b + "USDT" for b in BASES)
COLS = ["ts", "venue", "symbol", "side", "price", "qty", "usd"]
BINANCE_WS = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"
OKX_URL = "https://www.okx.com/api/v5/public/liquidation-orders"
OKX_INST = "https://www.okx.com/api/v5/public/instruments"
OKX_CTVAL = {"BTC": 0.01, "ETH": 0.1, "SOL": 1.0, "BNB": 0.01, "NEAR": 10.0, "FET": 10.0, "ADA": 100.0, "XRP": 100.0}
FLUSH_S = 60
OKX_POLL_S = 60
log = logging.getLogger("liq_recorder")


# ------------------------------------------------------------------ normalization (pure)
def _row(ts_ms, venue, symbol, side, price, qty):
    price, qty = float(price), float(qty)
    return dict(ts=int(ts_ms), venue=venue, symbol=symbol, side=side, price=price, qty=qty,
                usd=price * qty)


def norm_binance(msg: dict, symbols=SYMBOLS) -> list[dict]:
    """forceOrder: o.S is the ORDER side; SELL closes a long => long liquidated."""
    o = msg.get("o") if isinstance(msg, dict) else None
    if not o or o.get("s") not in symbols:
        return []
    price = float(o.get("ap") or 0) or float(o["p"])
    qty = float(o.get("z") or 0) or float(o["q"])
    if qty <= 0 or price <= 0:
        return []
    return [_row(o["T"], "binance", o["s"], "long" if o["S"] == "SELL" else "short", price, qty)]


def norm_bybit(msg: dict, symbols=SYMBOLS) -> list[dict]:
    """allLiquidation.<SYM>: S='Buy' means a LONG position was liquidated; v base-coin size, p bankruptcy px."""
    if not isinstance(msg, dict) or not str(msg.get("topic", "")).startswith("allLiquidation."):
        return []
    out = []
    for d in msg.get("data") or []:
        if d.get("s") not in symbols:
            continue
        qty, price = float(d["v"]), float(d["p"])
        if qty > 0 and price > 0:
            out.append(_row(d["T"], "bybit", d["s"], "long" if d["S"] == "Buy" else "short", price, qty))
    return out


def norm_okx(resp: dict, ctval: dict | None = None) -> list[dict]:
    """liquidation-orders: details[].side is the closing order side (sell => long liquidated);
    sz in contracts * ctVal = base coin; bkPx = bankruptcy price."""
    ctval = ctval or OKX_CTVAL
    out = []
    for blk in (resp or {}).get("data") or []:
        parts = blk.get("instId", "").split("-")           # e.g. BTC-USDT-SWAP
        if len(parts) != 3 or parts[1] != "USDT" or parts[0] not in ctval:
            continue
        sym = parts[0] + "USDT"
        for d in blk.get("details") or []:
            px, sz = float(d["bkPx"] or 0), float(d["sz"] or 0) * ctval[parts[0]]
            if px > 0 and sz > 0:
                out.append(_row(d["ts"], "okx", sym, "long" if d["side"] == "sell" else "short", px, sz))
    return out


def key(r: dict) -> tuple:
    return (r["venue"], r["symbol"], r["ts"], r["side"], round(r["price"], 8), round(r["qty"], 8))


# ------------------------------------------------------------------ storage
class Store:
    def __init__(self, root: Path = OUT):
        self.root = Path(root)
        self.buf: list[dict] = []
        self.seen: set = set()
        self._warm()

    def _warm(self):
        """Seed dedupe keys from the last two day files (restart-safe for OKX re-polls)."""
        days = [(pd.Timestamp.now("UTC") - pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (0, 1)]
        for vd in self.root.glob("*"):
            for d in days:
                p = vd / f"{d}.parquet"
                if p.exists():
                    try:
                        for r in pd.read_parquet(p).to_dict("records"):
                            r["ts"] = int(pd.Timestamp(r["ts"]).timestamp() * 1000)
                            self.seen.add(key(r))
                    except Exception as e:
                        log.warning("warm %s: %s", p, e)

    def add(self, rows: list[dict]) -> int:
        n = 0
        for r in rows:
            k = key(r)
            if k not in self.seen:
                self.seen.add(k)
                self.buf.append(r)
                n += 1
        return n

    def flush(self) -> int:
        if not self.buf:
            return 0
        rows, self.buf = self.buf, []
        df = pd.DataFrame(rows, columns=COLS)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
        try:
            for (venue, day), g in df.groupby(["venue", "day"]):
                p = self.root / venue / f"{day}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                g = g[COLS]
                if p.exists():
                    g = pd.concat([pd.read_parquet(p), g], ignore_index=True)
                g = g.drop_duplicates(["ts", "symbol", "side", "price", "qty"]).sort_values("ts")
                tmp = p.with_suffix(".tmp")
                g.to_parquet(tmp, index=False)
                os.replace(tmp, p)
        except Exception:
            self.buf = rows + self.buf       # retry next flush
            raise
        if len(self.seen) > 500_000:         # bound the dedupe set
            self.seen = {key(r) for r in rows}
        return len(rows)


def read_events(venue: str, start, end, root: Path = OUT) -> pd.DataFrame:
    """Load recorded rows for a venue within [start, end] (UTC timestamps)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    parts = []
    for d in pd.date_range(start.floor("D"), end.floor("D"), freq="D"):
        p = Path(root) / venue / f"{d.strftime('%Y-%m-%d')}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(parts, ignore_index=True)
    return df[(df["ts"] >= start) & (df["ts"] <= end)].reset_index(drop=True)


# ------------------------------------------------------------------ feeds
async def _ws_loop(name, url, on_open, handle, store: Store, ping_s: float | None = None):
    """ping_s=None -> protocol-level pings; else app-level {"op":"ping"} every ping_s of silence."""
    import websockets
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20 if ping_s is None else None,
                                          ping_timeout=30, max_queue=256, close_timeout=5) as ws:
                log.info("%s connected", name)
                backoff = 1.0
                if on_open:
                    await on_open(ws)
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=ping_s or 120)
                    except asyncio.TimeoutError:
                        if ping_s:
                            await ws.send(json.dumps({"op": "ping"}))
                            continue
                        # a quiet liquidation feed is normal; the protocol ping detects dead links
                        continue
                    store.add(handle(json.loads(raw)))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("%s dropped (%s: %s); retry in %.0fs", name, type(e).__name__, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def binance_feed(store):
    await _ws_loop("binance", BINANCE_WS, None, norm_binance, store)


async def bybit_feed(store):
    async def on_open(ws):
        await ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{s}" for s in SYMBOLS]}))
    await _ws_loop("bybit", BYBIT_WS, on_open, norm_bybit, store, ping_s=20)


def _okx_fetch(session, base: str) -> dict:
    r = session.get(OKX_URL, params=dict(instType="SWAP", state="filled", uly=f"{base}-USDT"), timeout=15)
    r.raise_for_status()
    return r.json()


async def okx_feed(store):
    sess = requests.Session()
    sess.headers["User-Agent"] = "bot_trade-liq-recorder/1.0"
    ctval = dict(OKX_CTVAL)
    try:
        r = await asyncio.to_thread(lambda: sess.get(OKX_INST, params=dict(instType="SWAP"), timeout=20).json())
        for d in r.get("data", []):
            p = d["instId"].split("-")
            if len(p) == 3 and p[1] == "USDT" and p[0] in ctval:
                ctval[p[0]] = float(d["ctVal"]) * float(d.get("ctMult") or 1)
    except Exception as e:
        log.warning("okx instruments failed, using defaults: %s", e)
    while True:
        t0 = time.monotonic()
        for b in BASES:
            try:
                store.add(norm_okx(await asyncio.to_thread(_okx_fetch, sess, b), ctval))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("okx %s: %s", b, e)
            await asyncio.sleep(0.5)
        await asyncio.sleep(max(1.0, OKX_POLL_S - (time.monotonic() - t0)))


async def flusher(store: Store):
    while True:
        await asyncio.sleep(FLUSH_S)
        try:
            n = store.flush()
            if n:
                log.info("flushed %d rows", n)
        except Exception as e:
            log.error("flush failed: %s", e)


async def main():
    store = Store()
    tasks = [asyncio.create_task(c(store)) for c in (binance_feed, bybit_feed, okx_feed)]
    tasks.append(asyncio.create_task(flusher(store)))
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        store.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
