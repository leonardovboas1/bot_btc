"""No-lookahead tests: computing on data[:k] must reproduce the full-data values at < k."""
import numpy as np
import pandas as pd
import pytest

from btclab.indicators import base as ta
from btclab.indicators.registry import REGISTRY, compute

from conftest import synthetic_ohlcv

CUTS = (157, 611, 1003, 1499)


def _same(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == bool:
        assert np.array_equal(a, b), name
    else:
        assert np.array_equal(a, b, equal_nan=True), name


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registry_indicator_causal(name, ohlcv):
    df = ohlcv.iloc[:1500]
    full = compute(name, df)
    for k in CUTS:
        part = compute(name, df.iloc[:k])
        for key, arr in full.items():
            assert len(part[key]) == k
            _same(part[key], np.asarray(arr)[:k], f"{name}.{key} @k={k}")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registry_indicator_future_perturbation(name, ohlcv):
    """Scrambling bars >= k must not change any value < k."""
    df = ohlcv.iloc[:1500]
    k = 900
    full = compute(name, df)
    rng = np.random.default_rng(1)
    pert = df.copy()
    scale = rng.uniform(0.7, 1.3, len(df) - k)
    for c in ("open", "high", "low", "close"):
        pert.iloc[k:, pert.columns.get_loc(c)] = df[c].to_numpy()[k:] * scale
    pert.iloc[k:, pert.columns.get_loc("volume")] = df["volume"].to_numpy()[k:][::-1]
    other = compute(name, pert)
    for key in full:
        _same(other[key][:k], full[key][:k], f"{name}.{key}")


@pytest.mark.parametrize("fn", [
    lambda d: ta.ema(d.close, 10), lambda d: ta.rma(d.close, 10), lambda d: ta.rsi(d.close, 14),
    lambda d: ta.hma(d.close, 21), lambda d: ta.atr(d.high, d.low, d.close, 14),
    lambda d: ta.stoch(d.close, d.high, d.low, 14), lambda d: ta.pivothigh(d.high, 5, 5),
    lambda d: ta.pivotlow(d.low, 5, 5),
    lambda d: ta.valuewhen(ta.crossover(d.close, ta.sma(d.close, 20)), d.close, 1),
    lambda d: ta.heikinashi(d.open, d.high, d.low, d.close)[0],
])
def test_base_primitives_causal(fn, synth):
    full = fn(synth)
    for k in CUTS:
        _same(fn(synth.iloc[:k]), full[:k], "base")


# ----------------------------------------------------------------------------- HTF
def test_htf_causal_truncation():
    df = synthetic_ohlcv(2000, freq="1h")
    f = lambda d: ta.ema(d["close"].to_numpy(), 5)  # noqa: E731
    for tf in ("240", "720", "1D", "W"):
        full = ta.request_security_htf(df, tf, f)
        for k in (50, 333, 1001, 1777, 1999):
            part = ta.request_security_htf(df.iloc[:k], tf, f)
            _same(part, full[:k], f"htf {tf} k={k}")


def test_htf_lookahead_off_semantics():
    """1h base, 4h HTF close: the 4h value appears on the 1h bar that completes the 4h
    bar (hour % 4 == 3) and is held for the next three bars."""
    df = synthetic_ohlcv(48, freq="1h", start="2024-01-01")
    out = ta.request_security_htf(df, "240", lambda d: d["close"].to_numpy())
    hi = ta.request_security_htf(df, "240", lambda d: d["high"].to_numpy())
    hours = df.index.hour
    c = df["close"].to_numpy()
    assert np.all(np.isnan(out[:3]))
    for i in range(3, 48):
        last_complete = i - ((hours[i] + 1) % 4)  # last 1h bar that closed a 4h bucket
        assert out[i] == c[last_complete]
        assert hi[i] == df["high"].to_numpy()[last_complete - 3:last_complete + 1].max()


def test_htf_never_uses_open_bucket():
    """Values seen at bar i must equal an HTF computation that only uses bars <= i."""
    df = synthetic_ohlcv(24 * 30, freq="1h")
    f = lambda d: d["close"].to_numpy()  # noqa: E731
    out = ta.request_security_htf(df, "1D", f)
    c = df["close"].to_numpy()
    for i in range(len(df)):
        hour = df.index[i].hour
        if hour == 23:
            assert out[i] == c[i]
        elif i >= 24:
            assert out[i] == c[i - hour - 1]


def test_htf_weekly_alignment_monday():
    df = synthetic_ohlcv(24 * 21, freq="1h", start="2024-01-03")  # Wednesday
    out = ta.request_security_htf(df, "W", lambda d: d["close"].to_numpy())
    c = df["close"].to_numpy()
    idx = df.index
    # the week starting Mon 2024-01-01 closes after the Sunday 23:00 bar -> visible from it on
    sun = np.flatnonzero((idx.dayofweek == 6) & (idx.hour == 23))
    assert np.all(np.isnan(out[:sun[0]]))
    assert out[sun[0]] == c[sun[0]]
    assert out[sun[0] + 5] == c[sun[0]]


def test_htf_bool_and_dict_results():
    df = synthetic_ohlcv(100, freq="1h")
    res = ta.request_security_htf(df, "240", lambda d: {"up": (d.close > d.open).to_numpy(),
                                                       "c": d.close.to_numpy()})
    assert res["up"].dtype == bool and not res["up"][:3].any()
    assert np.isnan(res["c"][:3]).all()
