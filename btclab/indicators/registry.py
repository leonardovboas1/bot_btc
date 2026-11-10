"""Indicator registry: name -> function, defaults, search space, emitted primitives.

Usage::

    from btclab.indicators.registry import REGISTRY, compute, all_primitives
    out = compute("gc", df_4h, period=100)          # dict of arrays
    out["filter_rising"]                             # bool array
    REGISTRY["gc"].primitives                        # ['gc.filter_rising', ...]

``df`` is a base-timeframe OHLCV DataFrame (DatetimeIndex or 'open_time' column).
Search-space entries:
    {"type": "int",   "low": a, "high": b, "step": s}
    {"type": "float", "low": a, "high": b, "step": s}
    {"type": "cat",   "choices": [...]}
``constraint(params) -> bool`` rejects invalid combinations (e.g. fast >= slow).
``warmup(params) -> int`` is the number of leading bars the signal library should skip
(important for zero-initialised recursions such as the Gaussian filter and STC).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import cipher_b as _cb
from . import gaussian_channel as _gc
from . import momentum as _mo
from . import stoch_rsi as _sr
from . import trend as _tr
from .base import times_ms


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    func: Callable
    inputs: tuple            # OHLCV column names (+ "time") passed positionally / by keyword
    defaults: dict
    space: dict
    primitives: tuple        # "<name>.<output key>" — bool outputs
    warmup: Callable = field(default=lambda p: 0)
    constraint: Callable = field(default=lambda p: True)
    description: str = ""

    @property
    def outputs(self) -> tuple:
        return tuple(p.split(".", 1)[1] for p in self.primitives)


def _i(lo, hi, step=1):
    return {"type": "int", "low": lo, "high": hi, "step": step}


def _f(lo, hi, step):
    return {"type": "float", "low": lo, "high": hi, "step": step}


def _c(*choices):
    return {"type": "cat", "choices": list(choices)}


def _prims(name, keys):
    return tuple(f"{name}.{k}" for k in keys)


def _run_gc(high, low, close, **p):
    return _gc.gaussian_channel(high, low, close, **p)


def _run_cipher(open, high, low, close, time, **p):
    return _cb.cipher_b(open, high, low, close, time=time, **p)


REGISTRY: dict[str, IndicatorSpec] = {}


def register(spec: IndicatorSpec) -> IndicatorSpec:
    REGISTRY[spec.name] = spec
    return spec


register(IndicatorSpec(
    name="gc",
    func=_run_gc,
    inputs=("high", "low", "close"),
    defaults=dict(poles=4, period=144, mult=1.414, mode_lag=False, mode_fast=False),
    space=dict(poles=_i(1, 6), period=_i(30, 200, 2), mult=_f(1.0, 2.5, 0.1),
               mode_lag=_c(False, True), mode_fast=_c(False, True)),
    primitives=_prims("gc", [
        "filter_rising", "filter_falling", "close_above_hband", "close_below_lband",
        "close_above_filt", "close_below_filt", "cross_above_hband", "cross_below_hband",
        "cross_above_filt", "cross_below_filt", "cross_above_lband", "cross_below_lband"]),
    warmup=lambda p: _gc.warmup_bars(**p),
    description="Gaussian Channel [DW] (catalog #6)",
))

register(IndicatorSpec(
    name="cipherb",
    func=_run_cipher,
    inputs=("open", "high", "low", "close", "time"),
    defaults=dict(wt_channel_len=9, wt_average_len=12, wt_ma_len=3, ob_level=53.0, os_level=-53.0,
                  rsi_mfi_period=60, rsi_len=14),
    space=dict(wt_channel_len=_i(6, 14), wt_average_len=_i(9, 21), wt_ma_len=_i(2, 6),
               ob_level=_f(45.0, 70.0, 1.0), os_level=_f(-70.0, -45.0, 1.0),
               rsi_mfi_period=_i(30, 100, 5), rsi_len=_i(7, 21)),
    primitives=_prims("cipherb", [
        "buy", "sell", "gold_buy", "bull_div", "bear_div",
        "wt_bull_div", "wt_bear_div", "wt_bull_div_add", "wt_bear_div_add",
        "wt_bull_div_hidden", "wt_bear_div_hidden",
        "rsi_bull_div", "rsi_bear_div", "rsi_bull_div_hidden", "rsi_bear_div_hidden",
        "stoch_bull_div", "stoch_bear_div", "stoch_bull_div_hidden", "stoch_bear_div_hidden",
        "wt_cross_up", "wt_cross_down", "wt_oversold", "wt_overbought",
        "wt_above_zero", "wt_below_zero", "mfi_green", "mfi_red",
        "stoch_k_cross_up_d", "stoch_k_cross_down_d", "stc_cross_up_25", "stc_cross_down_75",
        "sommi_bull_flag", "sommi_bear_flag", "sommi_bull_diamond", "sommi_bear_diamond"]),
    # slowest pieces: STC (ema 50 + 2 windows, zero-initialised) and the 12h HTF WaveTrend
    warmup=lambda p: max(200, 3 * 30 * 3),
    description="VuManChu Cipher B + Divergences (catalog #1, #2, #21); HTF lookahead OFF",
))

register(IndicatorSpec(
    name="stochrsi",
    func=lambda close, **p: _sr.stoch_rsi(close, **p),
    inputs=("close",),
    defaults=dict(rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3, ob=80.0, os=20.0, use_log=False),
    space=dict(rsi_len=_i(7, 21), stoch_len=_i(7, 21), smooth_k=_i(2, 5), smooth_d=_i(2, 5),
               ob=_f(70.0, 90.0, 5.0), os=_f(10.0, 30.0, 5.0), use_log=_c(False, True)),
    primitives=_prims("stochrsi", [
        "k_cross_up_d", "k_cross_down_d", "bull_cross_os", "bear_cross_ob", "k_above_ob",
        "k_below_os", "k_cross_up_os", "k_cross_down_ob", "k_above_50",
        "k_ge_ob", "k_le_os", "k_extreme"]),
    warmup=lambda p: 3 * (p["rsi_len"] + p["stoch_len"] + p["smooth_k"] + p["smooth_d"]),
    description="Stochastic RSI (catalog #5)",
))

register(IndicatorSpec(
    name="macd",
    func=lambda close, **p: _mo.macd(close, **p),
    inputs=("close",),
    defaults=dict(fast=12, slow=26, signal=9),
    space=dict(fast=_i(5, 20), slow=_i(20, 60), signal=_i(3, 15)),
    primitives=_prims("macd", [
        "cross_up", "cross_down", "cross_up_below_zero", "cross_down_above_zero",
        "above_zero", "below_zero", "hist_rising", "zero_cross_up", "zero_cross_down"]),
    warmup=lambda p: 3 * (p["slow"] + p["signal"]),
    constraint=lambda p: p["fast"] < p["slow"],
    description="MACD (catalog #7)",
))

register(IndicatorSpec(
    name="cci",
    func=lambda high, low, close, **p: _mo.cci(high, low, close, **p),
    inputs=("high", "low", "close"),
    defaults=dict(length=20, level=100.0),
    space=dict(length=_i(10, 50), level=_f(80.0, 200.0, 10.0)),
    primitives=_prims("cci", ["cross_up_os", "cross_down_ob", "cross_up_ob", "cross_down_os",
                              "above_zero", "below_zero"]),
    warmup=lambda p: p["length"],
    description="CCI (catalog #22)",
))

register(IndicatorSpec(
    name="willr",
    func=lambda high, low, close, **p: _mo.williams_r(high, low, close, **p),
    inputs=("high", "low", "close"),
    defaults=dict(length=14, os=-80.0, ob=-20.0),
    space=dict(length=_i(5, 30), os=_f(-90.0, -70.0, 5.0), ob=_f(-30.0, -10.0, 5.0)),
    primitives=_prims("willr", ["cross_up_os", "cross_down_ob", "below_os", "above_ob"]),
    warmup=lambda p: p["length"],
    description="Williams %R (catalog #23)",
))

register(IndicatorSpec(
    name="mfi",
    func=lambda high, low, close, volume, **p: _mo.mfi(high, low, close, volume, **p),
    inputs=("high", "low", "close", "volume"),
    defaults=dict(length=14, os=20.0, ob=80.0),
    space=dict(length=_i(7, 30), os=_f(10.0, 30.0, 5.0), ob=_f(70.0, 90.0, 5.0)),
    primitives=_prims("mfi", ["cross_up_os", "cross_down_ob", "below_os", "above_ob"]),
    warmup=lambda p: p["length"],
    description="Money Flow Index (catalog #24)",
))

register(IndicatorSpec(
    name="supertrend",
    func=lambda high, low, close, **p: _tr.supertrend(high, low, close, **p),
    inputs=("high", "low", "close"),
    defaults=dict(atr_len=10, factor=3.0),
    space=dict(atr_len=_i(7, 30), factor=_f(1.5, 5.0, 0.25)),
    primitives=_prims("supertrend", ["up", "down", "flip_up", "flip_down"]),
    warmup=lambda p: 5 * p["atr_len"],
    description="Supertrend (catalog #8)",
))

register(IndicatorSpec(
    name="dmi",
    func=lambda high, low, close, **p: _tr.dmi(high, low, close, **p),
    inputs=("high", "low", "close"),
    defaults=dict(di_len=14, adx_len=14, adx_thr=25.0),
    space=dict(di_len=_i(7, 30), adx_len=_i(7, 30), adx_thr=_f(15.0, 35.0, 1.0)),
    primitives=_prims("dmi", ["di_cross_up", "di_cross_down", "bull", "bear", "adx_strong",
                              "adx_rising", "bull_strong", "bear_strong"]),
    warmup=lambda p: 5 * (p["di_len"] + p["adx_len"]),
    description="ADX / DMI (catalog #13)",
))

register(IndicatorSpec(
    name="psar",
    func=lambda high, low, close, **p: _tr.psar(high, low, close, **p),
    inputs=("high", "low", "close"),
    defaults=dict(start=0.02, inc=0.02, maximum=0.2),
    space=dict(start=_f(0.005, 0.05, 0.005), inc=_f(0.005, 0.05, 0.005), maximum=_f(0.1, 0.4, 0.05)),
    primitives=_prims("psar", ["long", "short", "flip_long", "flip_short"]),
    warmup=lambda p: 50,
    constraint=lambda p: p["start"] <= p["maximum"],
    description="Parabolic SAR (catalog #14)",
))

register(IndicatorSpec(
    name="macross",
    func=lambda close, **p: _tr.ma_cross(close, **p),
    inputs=("close",),
    defaults=dict(fast=20, slow=50, kind="ema"),
    space=dict(fast=_i(5, 100), slow=_i(20, 300, 5), kind=_c("ema", "sma")),
    primitives=_prims("macross", ["fast_above_slow", "fast_below_slow", "cross_up", "cross_down",
                                  "close_above_slow", "close_below_slow", "close_above_fast"]),
    warmup=lambda p: 3 * p["slow"] if p["kind"] in ("ema", "rma") else p["slow"],
    constraint=lambda p: p["fast"] < p["slow"],
    description="EMA/SMA crosses (catalog #15)",
))


# ----------------------------------------------------------------------------- reference strategies
# brayvb strategy v2 (research/pine/brayvb_strategy_v2.pine, Pine v6, user-supplied):
#   isGreenChannel = filt > filt[1]; isPriceAboveHighBand = close > hband
#   isStochExtreme = k >= 80 or k <= 20
#   long  = isGreenChannel and isPriceAboveHighBand and isStochExtreme and timeCondition
#   close = ta.crossunder(close, hband) and timeCondition
# Pine strategy defaults: orders fill at the NEXT bar open, 100 % equity, commission 0.1 %,
# pyramiding 1, long only, timeCondition from 2018-01-01 00:00 UTC.
BRAYVB_V2_DEFAULTS = dict(poles=4, period=144, mult=1.414, mode_lag=False, mode_fast=False,
                          rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3, ob=80.0, os=20.0)


def brayvb_v2(high, low, close, **p) -> dict:
    p = {**BRAYVB_V2_DEFAULTS, **p}
    g = _gc.gaussian_channel(high, low, close, poles=p["poles"], period=p["period"], mult=p["mult"],
                             mode_lag=p["mode_lag"], mode_fast=p["mode_fast"])
    s = _sr.stoch_rsi(close, rsi_len=p["rsi_len"], stoch_len=p["stoch_len"], smooth_k=p["smooth_k"],
                      smooth_d=p["smooth_d"], ob=p["ob"], os=p["os"])
    entry = g["filter_rising"] & g["close_above_hband"] & s["k_extreme"]
    return {"long_entry": entry, "long_exit": g["cross_below_hband"],
            "filt": g["filt"], "hband": g["hband"], "k": s["k"], "d": s["d"]}


register(IndicatorSpec(
    name="brayvb_v2",
    func=brayvb_v2,
    inputs=("high", "low", "close"),
    defaults=dict(BRAYVB_V2_DEFAULTS),
    space=dict(poles=_i(1, 6), period=_i(30, 200, 2), mult=_f(1.0, 2.5, 0.1),
               ob=_f(70.0, 90.0, 5.0), os=_f(10.0, 30.0, 5.0)),
    primitives=_prims("brayvb_v2", ["long_entry", "long_exit"]),
    warmup=lambda p: _gc.warmup_bars(poles=p["poles"], period=p["period"], mode_lag=p["mode_lag"]),
    description="Reference strategy: long gc.filter_rising & close>hband & (k>=80 | k<=20); "
                "exit crossunder(close, hband)",
))

# Strategy metadata (DSL-style) for the backtester / Pine emitter.
STRATEGIES = {
    "brayvb_v2": dict(
        source="research/pine/brayvb_strategy_v2.pine",
        side="long",
        entry={"all": ["gc.filter_rising", "gc.close_above_hband",
                       {"any": ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]}]},
        exit={"any": ["gc.cross_below_hband"]},
        params={"gc": dict(poles=4, period=144, mult=1.414, mode_lag=False, mode_fast=False),
                "stochrsi": dict(rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3, ob=80.0, os=20.0)},
        execution=dict(fill="next_open", commission_pct=0.1, qty_pct_equity=100, pyramiding=1,
                       start="2018-01-01T00:00:00Z", initial_capital=100000),
        precomputed="brayvb_v2",   # REGISTRY entry emitting long_entry / long_exit
    ),
}


# ----------------------------------------------------------------------------- helpers
def _column(data, name):
    if name == "time":
        return times_ms(data) if hasattr(data, "columns") else np.asarray(data["time"])
    if hasattr(data, "columns"):
        return data[name].to_numpy(dtype=np.float64)
    return np.asarray(data[name], dtype=np.float64)


def compute(name: str, data, **params) -> dict:
    """Run indicator ``name`` on ``data`` (DataFrame or dict of arrays) with defaults
    overridden by ``params``."""
    spec = REGISTRY[name]
    p = {**spec.defaults, **params}
    args = {k: _column(data, k) for k in spec.inputs}
    return spec.func(**args, **p)


def all_primitives() -> list[str]:
    return [p for s in REGISTRY.values() for p in s.primitives]


def warmup(name: str, **params) -> int:
    spec = REGISTRY[name]
    return int(spec.warmup({**spec.defaults, **params}))
