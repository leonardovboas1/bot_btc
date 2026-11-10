import time

from src.data import hl_collector as hc
from btclab.liquidity import hl_levels as hl

A1, A2 = "0x" + "a" * 40, "0x" + "B" * 40

STATE = {"assetPositions": [
    {"type": "oneWay", "position": {"coin": "BTC", "szi": "0.5", "entryPx": "100000", "positionValue": "50000",
                                    "leverage": {"type": "cross", "value": 10}, "liquidationPx": "91000", "marginUsed": "5000"}},
    {"type": "oneWay", "position": {"coin": "ETH", "szi": "-10", "entryPx": "3000", "positionValue": "30000",
                                    "leverage": {"type": "isolated", "value": 5}, "liquidationPx": "3500.5", "marginUsed": "6000"}},
    {"type": "oneWay", "position": {"coin": "DOGE", "szi": "100", "positionValue": "10"}},
]}
ORDERS = [
    {"coin": "BTC", "side": "A", "limitPx": "90000", "sz": "0.5", "oid": 1, "isTrigger": True, "triggerPx": "90000",
     "reduceOnly": True, "orderType": "Stop Market"},
    {"coin": "BTC", "side": "B", "limitPx": "99000", "sz": "1", "oid": 2, "isTrigger": False},
    {"coin": "BTC", "side": "B", "sz": "0.0", "origSz": "2.0", "oid": 3, "isTrigger": True, "triggerPx": "110000",
     "isPositionTpsl": True, "orderType": "Take Profit Market"},
]


def test_parse_trades():
    msg = {"channel": "trades", "data": [{"coin": "BTC", "users": [A1, A2]}, {"users": ["bad"]}]}
    assert hc.parse_trades(msg) == {A1, A2.lower()}
    assert hc.parse_trades({"channel": "subscriptionResponse"}) == set()


def test_parse_state_and_orders():
    rows = hc.parse_state(STATE, now=1.0)
    assert [(r["coin"], r["side"], r["liq_px"], r["notional"], r["leverage"]) for r in rows] == [
        ("BTC", "long", 91000.0, 50000.0, 10.0), ("ETH", "short", 3500.5, 30000.0, 5.0)]
    o = hc.parse_orders(ORDERS, now=1.0)
    assert [(x["oid"], x["side"], x["trigger_px"], x["size"], x["reduce_only"]) for x in o] == [
        (1, "sell", 90000.0, 0.5, 1), (3, "buy", 110000.0, 2.0, 1)]


def test_parse_ctxs():
    r = hc.parse_ctxs(({"universe": [{"name": "BTC"}, {"name": "XYZ"}]},
                       [{"openInterest": "10", "markPx": "100000"}, {"openInterest": "1", "markPx": "1"}]), now=1.0)
    assert r == [dict(coin="BTC", oi=10.0, mark=100000.0, oi_usd=1_000_000.0, updated_at=1.0)]


def test_store_and_aggregate(tmp_path):
    p = tmp_path / "hl.sqlite"
    db = hc.connect(p)
    now = time.time()
    hc.add_addresses(db, [A1, A2.lower()], now)
    hc.save_ctxs(db, hc.parse_ctxs(({"universe": [{"name": "BTC"}]}, [{"openInterest": "10", "markPx": "100000"}]), now=now))
    hc.save_state(db, A1, hc.parse_state(STATE, now=now - 120), now)
    hc.save_orders(db, A1, hc.parse_orders(ORDERS, now=now - 60), now)
    assert hc.next_state_addr(db)[0] == A2.lower()  # never polled first
    ro = hl.open_db(p)
    d = hl.liq_levels("BTC", 100, db=ro, now=now)
    assert not d["empty"] and len(d["levels"]) == 1
    lv = d["levels"][0]
    assert lv["lo"] <= 91000 < lv["hi"] and lv["long_usd"] == 50000 and lv["n_long"] == 1 and lv["short_usd"] == 0
    assert abs(lv["age_long_s"] - 120) < 1
    c = d["coverage"]
    assert c["addresses"] == 2 and c["positions"] == 1 and abs(c["oi_share"] - 0.05) < 1e-9
    s = hl.stops("BTC", 100, db=ro, now=now)
    sell = [x for x in s["levels"] if x["sell_usd"]][0]
    assert sell["sell_usd"] == 90000 * 0.5 and sell["n_sell"] == 1
    assert hl.coin_of("BTCUSDT") == "BTC" and hl.coin_of("NEAR") == "NEAR"
    assert hl.liq_levels("ADA", 100, db=ro, now=now)["empty"]


def test_limiter_rate():
    import asyncio
    lim = hc.Limiter(per_min=6000)  # 100/s

    async def go():
        t = time.monotonic()
        for _ in range(5):
            await lim.spend(10)
        return time.monotonic() - t
    assert asyncio.run(go()) >= 0.3
