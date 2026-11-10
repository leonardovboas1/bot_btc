import pandas as pd
import pytest
from fastapi.testclient import TestClient

from btclab.api.app import app
from btclab.data import loaders

c = TestClient(app)


def test_base_paths():
    assert loaders._base("BTCUSDT") == loaders.RAW
    assert loaders._base("ethusdt") == loaders.RAW / "ETHUSDT"
    assert loaders._kline_path("perp", "1h", "ETHUSDT") == loaders.RAW / "ETHUSDT" / "um_futures" / "klines_1h.parquet"
    assert loaders._kline_path("spot", "1d") == loaders.RAW / "spot" / "klines_1d.parquet"


def test_list_symbols():
    syms = loaders.list_symbols()
    assert syms[0] == "BTCUSDT"


def test_resample_week():
    idx = pd.date_range("2024-01-03", periods=21, freq="D", tz="UTC")  # Wed start
    df = pd.DataFrame({"open": range(21), "high": range(21), "low": range(21),
                       "close": range(21), "volume": 1.0}, index=idx)
    w = loaders._resample_week(df)
    assert len(w) == 2  # partial first/last weeks dropped
    assert all(t.dayofweek == 0 for t in w.index)
    assert w.index[0] == pd.Timestamp("2024-01-08", tz="UTC")
    assert w["volume"].iloc[0] == 7 and w["open"].iloc[0] == 5 and w["close"].iloc[0] == 11


def test_load_1w_btc():
    w = loaders.load_ohlcv("perp", "1w", allow_holdout=True)
    assert len(w) > 100 and (w.index.dayofweek == 0).all()


def test_symbols_endpoint():
    r = c.get("/symbols").json()
    btc = r[0]
    assert btc["symbol"] == "BTCUSDT" and btc["base"] == "BTC"
    assert "1w" in btc["tfs"] and btc["has_liq"] is True


def test_ohlcv_eth():
    if "ETHUSDT" not in loaders.list_symbols():
        pytest.skip("ETHUSDT not downloaded")
    r = c.get("/ohlcv", params={"symbol": "ETHUSDT", "tf": "1h"}).json()
    assert r["symbol"] == "ETHUSDT" and len(r["bars"]) > 1000
    btc = c.get("/ohlcv", params={"tf": "1h"}).json()
    assert r["bars"][0]["close"] != btc["bars"][0]["close"]


def test_unknown_symbol_missing():
    with pytest.raises(FileNotFoundError):
        loaders.load_ohlcv("perp", "1h", allow_holdout=True, symbol="NOPEUSDT")


def test_heatmap_other_symbol():
    r = c.get("/heatmap/liq", params={"symbol": "ETHUSDT", "range": "1d", "end": "2026-11-01T00:00:00"})
    assert r.status_code in (200, 400)
