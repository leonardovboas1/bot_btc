import pytest
from fastapi.testclient import TestClient

from btclab.api.app import app

c = TestClient(app)


def test_ohlcv():
    r = c.get("/ohlcv", params={"tf": "4h"}).json()
    assert r["holdout_start"] > 0
    assert r["bars"][-1]["time"] >= r["holdout_start"]
    assert {"time", "open", "close"} <= set(r["bars"][0])
    r2 = c.get("/ohlcv", params={"tf": "4h", "start": "2024-01-01", "end": "2024-02-01"}).json()
    assert 0 < len(r2["bars"]) < 300


def test_strategies():
    assert any(x["name"] == "brayvb_v2" for x in c.get("/strategies").json())


@pytest.fixture(scope="module")
def bt():
    r = c.post("/backtest", json={"strategy": "brayvb_v2", "market": "perp", "tf": "4h",
                                  "fee": "binance_perp_taker"})
    assert r.status_code == 200, r.text
    return r.json()


def test_backtest_in_sample(bt):
    h = bt["holdout_start"]
    assert bt["trades"] and bt["metrics"]["n_trades"] == len(bt["trades"])
    assert all(t["exit_time"] < h for t in bt["trades"])
    assert all(p["time"] < h for p in bt["equity"])
    assert len(bt["equity"]) == len(bt["buy_hold"])
    assert all(m["time"] < h for m in bt["markers"])
    assert {m["text"] for m in bt["markers"]} <= {"LONG", "SHORT", "EXIT"}
    assert set(bt["trades"][0]) >= {"entry_time", "exit_time", "side", "pnl_pct", "exit_reason"}


def test_holdout_markers(bt):
    h = bt["holdout_start"]
    assert all(m["time"] >= h for m in bt["holdout_markers"])
    assert "holdout_metrics" not in bt


def test_backtest_errors():
    assert c.post("/backtest", json={"strategy": "nope"}).status_code == 404
    bad = {"name": "x", "tf": "4h", "indicators": {"gc": {}}, "long": ["gc.nope"]}
    assert c.post("/backtest", json={"spec": bad}).status_code == 400


def test_signal_current():
    r = c.get("/signal/current", params={"strategy": "brayvb_v2", "tf": "4h"}).json()
    assert r["state"] in ("LONG", "SHORT", "FLAT") and r["price"] > 0 and r["time"] > 0


def test_indicator():
    r = c.get("/indicator/gc", params={"tf": "4h", "params": "{}"}).json()
    assert r["series"] and all(len(v) > 0 for v in r["series"].values())
    assert c.get("/indicator/zzz").status_code == 404
    assert c.get("/indicator/gc", params={"params": "x"}).status_code == 400


def test_cors():
    r = c.options("/strategies", headers={"Origin": "http://localhost:5173",
                                          "Access-Control-Request-Method": "GET"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
