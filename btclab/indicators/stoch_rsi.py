"""Stochastic RSI (TradingView built-in "Stoch RSI").

    rsi1 = ta.rsi(src, lengthRSI)
    k = ta.sma(ta.stoch(rsi1, rsi1, rsi1, lengthStoch), smoothK)
    d = ta.sma(k, smoothD)

``use_log`` applies ``log(src)`` first (the Cipher B option).
"""
from __future__ import annotations

import numpy as np

from .base import crossover, crossunder, f64, rsi, sma, stoch


def stoch_rsi_kd(src, rsi_len: int = 14, stoch_len: int = 14, smooth_k: int = 3,
                 smooth_d: int = 3, use_log: bool = False):
    s = f64(src)
    if use_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.log(s)
    r = rsi(s, rsi_len)
    k = sma(stoch(r, r, r, stoch_len), smooth_k)
    d = sma(k, smooth_d)
    return k, d


def stoch_rsi(close, rsi_len: int = 14, stoch_len: int = 14, smooth_k: int = 3, smooth_d: int = 3,
              ob: float = 80.0, os: float = 20.0, use_log: bool = False) -> dict:
    k, d = stoch_rsi_kd(close, rsi_len, stoch_len, smooth_k, smooth_d, use_log)
    up = crossover(k, d)
    dn = crossunder(k, d)
    return {
        "k": k,
        "d": d,
        "k_cross_up_d": up,
        "k_cross_down_d": dn,
        "bull_cross_os": up & (k < os),
        "bear_cross_ob": dn & (k > ob),
        "k_above_ob": k > ob,
        "k_ge_ob": k >= ob,                       # brayvb v2: k >= 80
        "k_le_os": k <= os,                       # brayvb v2: k <= 20
        "k_extreme": (k >= ob) | (k <= os),       # brayvb v2 isStochExtreme
        "k_below_os": k < os,
        "k_cross_up_os": crossover(k, os),
        "k_cross_down_ob": crossunder(k, ob),
        "k_above_50": k > 50.0,
    }
