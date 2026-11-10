from fastapi.testclient import TestClient

from btclab.api.app import app

c = TestClient(app)


def test_fees():
    r = c.get("/fees").json()
    names = {f["name"] for f in r}
    assert {"binance_perp_taker", "spot", "hyperliquid_taker"} <= names
    assert all("fee_rate" in f for f in r)


def _shape_ok(r, keys):
    assert len(r["times"]) <= 400 and len(r["prices"]) <= 150
    for k in keys:
        assert len(r[k]) == len(r["times"])
        assert all(len(row) == len(r["prices"]) for row in r[k])


def test_book_in_range():
    r = c.get("/heatmap/book", params={"tf": "1h", "start": "2024-03-01", "end": "2024-03-10"})
    assert r.status_code == 200, r.text
    r = r.json()
    assert not r["empty"]
    _shape_ok(r, ["bid", "ask"])


def test_book_out_of_range():
    r = c.get("/heatmap/book", params={"tf": "1d", "start": "2019-01-01", "end": "2019-03-01"}).json()
    assert r["empty"] and r["available_from"] > 0


def test_liq():
    r = c.get("/heatmap/liq", params={"range": "1d", "end": "2024-03-20T00:00:00"})
    assert r.status_code == 200, r.text
    r = r.json()
    T, P = r["shape"]
    assert len(r["times"]) == T and len(r["prices"]) == P and 200 <= P <= 300
    import base64
    assert len(base64.b64decode(r["grid"])) == T * P and r["scale"] > 0
    assert {"active", "events", "candles"} <= set(r)


def test_volume_profile():
    r = c.get("/volume_profile", params={"tf": "1h", "start": "2024-03-01", "end": "2024-03-20"})
    assert r.status_code == 200, r.text
    r = r.json()
    assert len(r["prices"]) == len(r["volume"]) <= 150
    assert "poc" in r


def test_bad_tf():
    assert c.get("/heatmap/liq", params={"range": "7m"}).status_code == 400
