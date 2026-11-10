import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw" / "um_futures"


def synthetic_ohlcv(n=1500, freq="4h", seed=7, start="2021-01-01"):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, n) + 0.0002 * np.sin(np.arange(n) / 50.0)
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate(([close[0]], close[:-1])) * (1 + rng.normal(0, 0.001, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.lognormal(8, 0.5, n)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="open_time")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)


def load_real(tf, tail=None):
    f = RAW / f"klines_{tf}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)[["open", "high", "low", "close", "volume"]]
    return df.iloc[-tail:] if tail else df


@pytest.fixture(scope="session")
def ohlcv():
    """Real BTCUSDT perp 4h (last 2000 bars) if downloaded, else synthetic."""
    df = load_real("4h", tail=2000)
    return df if df is not None else synthetic_ohlcv(2000)


@pytest.fixture(scope="session")
def synth():
    return synthetic_ohlcv(1500)
