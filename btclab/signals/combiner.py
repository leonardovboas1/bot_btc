"""Run a DSL spec through the simulator and metrics."""
from __future__ import annotations

import json

import numpy as np

from btclab.engine import costs as _costs
from btclab.engine.metrics import compute_metrics_from_result
from btclab.engine.sim import simulate
from btclab.indicators import base as _base
from btclab.indicators.registry import REGISTRY, compute
from btclab.signals.dsl import RISK_KEYS, build_signal, validate_spec

ATR_LEN = 14


def _cost_model(name):
    for k, m in _costs.PRESETS.items():
        if str(name).lower() in (k.lower(), m.name.lower()):
            return m
    raise ValueError(f"unknown cost model {name!r}")


SOURCES = ("real", "ha")


def heikin_ashi(df):
    """Heikin Ashi OHLC (causal: bar t uses bars <= t). Other columns are copied unchanged."""
    from scipy.signal import lfilter
    o, h, l, c = (df[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    hc = (o + h + l + c) / 4.0
    ho = np.empty_like(hc)
    if len(hc):
        # ho[i] = 0.5*ho[i-1] + 0.5*hc[i-1], ho[0] = (o0+c0)/2
        x = np.concatenate(([(o[0] + c[0]) / 2.0], 0.5 * hc[:-1]))
        ho = lfilter([1.0], [1.0, -0.5], x)
    out = df.copy()
    out["open"], out["close"] = ho, hc
    out["high"] = np.maximum.reduce([h, ho, hc])
    out["low"] = np.minimum.reduce([l, ho, hc])
    return out


def _cached_compute(cache: dict, df, name: str, params: dict, source: str = "real"):
    full = {**REGISTRY[name].defaults, **(params or {})}
    key = (name, source, json.dumps(full, sort_keys=True, default=str))
    if key not in cache:
        cache[key] = compute(name, df, **(params or {}))
    return cache[key]


def run_spec(spec: dict, df, _cache: dict | None = None):
    """Returns ``(SimResult, metrics dict)``. ``df`` is base-tf OHLCV with a time index.
    If df has a ``funding`` column it is fed to the simulator."""
    validate_spec(spec)
    cache = {} if _cache is None else _cache
    n = len(df)
    inds = spec["indicators"]
    source = spec.get("source", "real")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; use one of {SOURCES}")
    # Signals read the chosen candles; fills, stops, ATR and costs always use real prices.
    sig_df = df
    if source == "ha":
        hk = ("__ha__", id(df), len(df))
        if hk not in cache:
            cache[hk] = heikin_ashi(df)
        sig_df = cache[hk]

    def lookup(prim):
        name, _, key = prim.partition(".")
        return _cached_compute(cache, sig_df, name, inds[name], source)[key]

    sigs = {s: build_signal(spec.get(s), lookup, n)
            for s in ("long", "short", "exit_long", "exit_short")}
    risk = {k: v for k, v in (spec.get("risk") or {}).items() if k in RISK_KEYS and v}
    kw = _cost_model(spec.get("costs", "BINANCE_PERP_TAKER")).sim_kwargs()
    if sigs["short"] is None:
        kw["allow_short"] = False

    def col(c):
        return df[c].to_numpy(dtype=np.float64)

    atr = _base.atr(col("high"), col("low"), col("close"), ATR_LEN)
    funding = col("funding") if "funding" in df.columns else None
    res = simulate(col("open"), col("high"), col("low"), col("close"),
                   sigs["long"], sigs["short"], sigs["exit_long"], sigs["exit_short"],
                   atr=atr, funding=funding, **kw, **risk)
    return res, compute_metrics_from_result(res, df.index)


def run_many(specs, df):
    """Run several specs sharing one in-memory indicator cache."""
    cache: dict = {}
    return [run_spec(s, df, cache) for s in specs]
