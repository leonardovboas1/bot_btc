import pandas as pd
from fastapi.testclient import TestClient

from src.data import gate, liq_recorder as R
from btclab.api.app import app


def test_norm_binance():
    m = {"e": "forceOrder", "o": {"s": "BTCUSDT", "S": "SELL", "q": "0.5", "p": "100", "ap": "101", "z": "0.4", "T": 1700000000123}}
    r = R.norm_binance(m)
    assert len(r) == 1 and r[0]["side"] == "long" and r[0]["price"] == 101 and r[0]["qty"] == 0.4
    assert abs(r[0]["usd"] - 40.4) < 1e-9
    m["o"]["S"] = "BUY"
    assert R.norm_binance(m)[0]["side"] == "short"
    m["o"]["s"] = "DOGEUSDT"
    assert R.norm_binance(m) == []


def test_norm_bybit():
    m = {"topic": "allLiquidation.ETHUSDT", "type": "snapshot", "ts": 1,
         "data": [{"T": 1700000000000, "s": "ETHUSDT", "S": "Buy", "v": "2", "p": "2000"},
                  {"T": 1700000000001, "s": "ETHUSDT", "S": "Sell", "v": "1", "p": "2001"},
                  {"T": 1, "s": "ETHUSD", "S": "Buy", "v": "1", "p": "1"}]}
    r = R.norm_bybit(m)
    assert [x["side"] for x in r] == ["long", "short"] and r[0]["usd"] == 4000
    assert R.norm_bybit({"op": "subscribe"}) == []


def test_norm_okx():
    resp = {"code": "0", "data": [{"instId": "ADA-USDT-SWAP", "details": [
        {"bkPx": "0.5", "posSide": "short", "side": "buy", "sz": "3", "ts": "1700000000000"},
        {"bkPx": "0.5", "posSide": "long", "side": "sell", "sz": "1", "ts": "1700000000001"}]},
        {"instId": "BTC-USD-SWAP", "details": [{"bkPx": "1", "side": "buy", "sz": "1", "ts": "1"}]}]}
    r = R.norm_okx(resp)
    assert [x["side"] for x in r] == ["short", "long"] and r[0]["qty"] == 300 and r[0]["usd"] == 150


def test_dedupe_and_flush(tmp_path):
    st = R.Store(tmp_path)
    row = R.norm_okx({"data": [{"instId": "BTC-USDT-SWAP", "details": [
        {"bkPx": "100", "side": "sell", "sz": "10", "ts": "1700000000000"}]}]})
    assert st.add(row) == 1 and st.add(row) == 0
    assert st.flush() == 1 and st.flush() == 0
    st2 = R.Store(tmp_path)               # restart: seen-keys not warmed for old days, parquet dedupe holds
    st2.add(row)
    st2.flush()
    df = R.read_events("okx", "2023-11-14", "2023-11-15", tmp_path)
    assert len(df) == 1 and df.iloc[0].symbol == "BTCUSDT" and df.iloc[0].side == "long"


def _gate_fixture(root):
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    idx = pd.date_range(t0, periods=4, freq="5min", name="ts")
    d = pd.DataFrame({c: 0.0 for c in gate.COLS}, index=idx)
    d["long_liq_usd"] = [0, 500.0, 0, 0]
    d["short_liq_usd"] = [0, 0, 0, 70.0]
    p = gate.path_for("BTCUSDT", root)
    p.parent.mkdir(parents=True)
    d.to_parquet(p)


def test_gate_rows_to_df():
    d = gate.rows_to_df([{"time": 1700000000, "long_liq_usd": 5, "open_interest_usd": 9},
                         {"time": 1700000000, "long_liq_usd": 6}])
    assert len(d) == 1 and d["long_liq_usd"].iloc[0] == 6 and set(gate.COLS) <= set(d.columns)


def test_endpoint(tmp_path, monkeypatch):
    liq = tmp_path / "liq"
    gdir = tmp_path / "gate"
    _gate_fixture(gdir)
    st = R.Store(liq)
    st.add([dict(ts=1704067500000, venue="bybit", symbol="BTCUSDT", side="long", price=42000.0, qty=0.1, usd=4200.0),
            dict(ts=1704067500000, venue="bybit", symbol="ETHUSDT", side="long", price=2000.0, qty=1, usd=2000.0)])
    st.flush()
    monkeypatch.setattr(R, "OUT", liq)
    monkeypatch.setattr(gate, "GATE_DIR", gdir)
    monkeypatch.setattr(gate.load, "__defaults__", (gdir,))
    monkeypatch.setattr(R.read_events, "__defaults__", (liq,))
    c = TestClient(app)
    r = c.get("/liquidations", params={"symbol": "btcusdt", "start": "2024-01-01", "end": "2024-01-02"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["events"]) == 1 and j["events"][0]["venue"] == "bybit" and j["events"][0]["usd"] == 4200
    assert [(b["long_usd"], b["short_usd"]) for b in j["bars"]] == [(500.0, 0.0), (0.0, 70.0)]
    r = c.get("/liquidations", params={"symbol": "BTCUSDT", "start": "2024-01-01", "end": "2024-01-02", "venues": "gate"})
    assert r.json()["events"] == [] and len(r.json()["bars"]) == 2
    assert c.get("/liquidations", params={"start": "2024-01-02", "end": "2024-01-01"}).status_code == 400
