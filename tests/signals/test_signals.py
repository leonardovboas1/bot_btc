import numpy as np
import pytest

from btclab.data import loaders
from btclab.signals import run_many, run_spec
from btclab.signals.dsl import SpecError, build_signal, validate_spec

A = np.array([1, 1, 0, 0], bool)
B = np.array([1, 0, 1, 0], bool)
LOOK = {"x.a": A, "x.b": B}.__getitem__


def test_and_or_not():
    assert (build_signal(["x.a", "x.b"], LOOK, 4) == [1, 0, 0, 0]).all()
    assert (build_signal([["x.a", "x.b"]], LOOK, 4) == [1, 1, 1, 0]).all()
    assert (build_signal(["not:x.a"], LOOK, 4) == [0, 0, 1, 1]).all()
    assert (build_signal(["x.a", ["x.b", "not:x.a"]], LOOK, 4) == [1, 0, 0, 0]).all()
    assert build_signal([], LOOK, 4) is None


def test_invalid_primitive():
    spec = {"name": "t", "tf": "4h", "indicators": {"gc": {}}, "long": ["gc.nope"]}
    with pytest.raises(SpecError):
        validate_spec(spec)
    with pytest.raises(SpecError):
        validate_spec({**spec, "long": ["bogus.x"]})
    with pytest.raises(SpecError):
        validate_spec({**spec, "long": ["macd.cross_up"]})  # not declared


def test_holdout_lock(monkeypatch):
    monkeypatch.delenv("HOLDOUT_UNLOCK", raising=False)
    with pytest.raises(loaders.HoldoutError):
        loaders.load_ohlcv("perp", "4h")
    df = loaders.load_ohlcv("perp", "4h", end=loaders.get_cutoff())
    assert df.index.max() < loaders.get_cutoff()
    assert len(loaders.load_ohlcv("perp", "4h", allow_holdout=True)) > len(df)
    monkeypatch.setenv("HOLDOUT_UNLOCK", "1")
    loaders.load_ohlcv("perp", "4h")


def test_cutoff_persisted():
    c = loaders.get_cutoff()
    assert loaders.HOLDOUT_FILE.exists() and loaders.get_cutoff() == c


def test_brayvb_v2_runs(monkeypatch):
    monkeypatch.delenv("HOLDOUT_UNLOCK", raising=False)
    end = loaders.get_cutoff()
    df = loaders.load_ohlcv("perp", "4h", end=end)
    df["funding"] = loaders.load_funding("4h", end=end).values
    spec = {"name": "brayvb_v2_dsl", "tf": "4h",
            "indicators": {"gc": {}, "stochrsi": {}},
            "long": ["gc.filter_rising", "gc.close_above_hband",
                     ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]],
            "exit_long": ["gc.cross_below_hband"],
            "risk": {}, "costs": "BINANCE_PERP_TAKER"}
    res, _ = run_spec(spec, df)
    assert res.n_trades > 0
    ((r2, _),) = run_many([spec], df)
    assert r2.n_trades == res.n_trades
