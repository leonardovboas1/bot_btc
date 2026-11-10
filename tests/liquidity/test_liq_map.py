import base64
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btclab.liquidity import liq_map  # noqa: E402

EDGES = (np.log(80.0), np.log(130.0 / 80.0) / 200, 200)


def series(n=60, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    high = close * 1.002
    low = close * 0.998
    oi = 1000 + np.cumsum(np.abs(rng.normal(5, 2, n)))
    return close, high, low, oi


def test_consumed_on_touch():
    close, high, low, oi = series()
    close[30:] = close[:30].mean()  # flat after bar 30
    high, low = close * 1.001, close * 0.999
    low = low.copy()
    g, L, S, ev, centers = liq_map.liq_columns(close, high, low, oi, 300, EDGES, 0)
    assert g[29].sum() > 0
    # crash bar 40 trades through everything below: every long level is gone from col 40 on
    low2 = low.copy(); low2[40] = 70.0
    g2, L2, S2, ev2, _ = liq_map.liq_columns(close, high, low2, oi, 300, EDGES, 0)
    assert any(e[0] == 40 and e[1] == "long" for e in ev2)
    # long levels created after the crash only (just below close) remain; none far below
    assert g2[40][centers < 0.85 * close[40]].sum() == 0
    # the same levels existed before in the un-crashed run
    assert g[40][centers < close[40]].sum() > 0 or L.sum() > 0


def test_causal():
    close, high, low, oi = series(80)
    full = liq_map.liq_columns(close, high, low, oi, 300, EDGES, 0)[0]
    cut = liq_map.liq_columns(close[:50], high[:50], low[:50], oi[:50], 300, EDGES, 0)[0]
    assert np.array_equal(full[:50], cut)


def test_active_levels_match_last_column():
    close, high, low, oi = series()
    g, L, S, ev, _ = liq_map.liq_columns(close, high, low, oi, 300, EDGES, 10)
    assert g.shape[0] == 50
    assert np.allclose(g[-1], (L + S).astype(np.float32), rtol=1e-5)


def test_compute_shape_offline(monkeypatch):
    import pandas as pd
    idx = pd.date_range("2024-03-01", periods=2 * 288 + 5 * 288, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    c = 60000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    k = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)
    oi = pd.Series(50000 + np.cumsum(rng.normal(2, 5, len(idx))), index=idx)
    monkeypatch.setattr(liq_map, "_binance_5m", lambda *a, **kw: (k, oi))
    d = liq_map.compute("BTCUSDT", "1d", end=idx[-1] + pd.Timedelta("5min"))
    T, P = d["shape"]
    assert len(base64.b64decode(d["grid"])) == T * P and 200 <= P <= 300 and T == len(d["candles"]["c"])
    assert d["resolution"] == "5min"
