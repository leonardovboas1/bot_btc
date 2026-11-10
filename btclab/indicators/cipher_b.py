"""VuManChu Cipher B + Divergences (Pine v4, MPL-2.0) — Pine-faithful port.

Components: WaveTrend (wt1/wt2/vwap, OB/OS, crosses, buy/sell circles), fractal
divergences (``f_findDivs``: WT with limits, WT "add" range, no-limit/hidden,
RSI, Stoch RSI), RSI+MFI area, RSI, Stoch RSI (log option), Schaff Trend Cycle,
gold buy, Sommi flags and diamonds.

Timing
* Divergence / fractal booleans are reported on the *confirmation* bar ``t``
  (pivot at ``t-2``). The original plots them with ``offset=-2``; using the plotted
  position in a backtest is lookahead.
* Higher-timeframe parts use :func:`base.request_security_htf` with lookahead OFF:
  a base bar sees an HTF value only once that HTF bar has closed.
  The ORIGINAL script requests the Heikin-Ashi HTF candles for the Sommi diamond
  with ``barmerge.lookahead_on`` (and no ``[1]`` offset). On historical bars that
  leaks the final HTF candle into every earlier base bar of the period, which is a
  repaint / lookahead bug in any backtest. We deliberately do not reproduce it, so
  diamonds can differ from TradingView's historical plot. The Sommi flag's HTF
  WaveTrend uses the v4 default (``lookahead_off``) and is reproduced.
* If an HTF is not larger than the chart timeframe (e.g. diamond '60' on a 4h
  chart) it is evaluated on the chart bars (Pine would read the last intrabar
  of a lower timeframe, which base data cannot reproduce).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (_div, _truthy, cross, crossover, crossunder, ema, f64, heikinashi, highest,
                   hlc3, lowest, request_security_htf, rsi, shift, sma, valuewhen)
from .stoch_rsi import stoch_rsi_kd

DEFAULTS = dict(
    wt_channel_len=9, wt_average_len=12, wt_ma_len=3,
    ob_level=53.0, ob_level2=60.0, ob_level3=100.0,
    os_level=-53.0, os_level2=-60.0, os_level3=-75.0,
    wt_div_ob_level=45.0, wt_div_os_level=-65.0,
    wt_div_ob_level_add=15.0, wt_div_os_level_add=-40.0,
    wt_show_div=True, show_hidden_div_nl=True,
    rsi_mfi_period=60, rsi_mfi_multiplier=150.0, rsi_mfi_pos_y=2.5,
    rsi_len=14, rsi_div_ob_level=60.0, rsi_div_os_level=30.0, rsi_show_div=False,
    stoch_use_log=True, stoch_avg=False, stoch_len=14, stoch_rsi_len=14,
    stoch_k_smooth=3, stoch_d_smooth=3, stoch_show_div=False,
    tc_length=10, tc_fast_length=23, tc_slow_length=50, tc_factor=0.5,
    sommi_vwap_tf="720", sommi_vwap_bear_level=0.0, sommi_vwap_bull_level=0.0,
    sommi_flag_wt_bear_level=0.0, sommi_flag_wt_bull_level=0.0,
    sommi_rsimfi_bear_level=0.0, sommi_rsimfi_bull_level=0.0,
    sommi_htc_res="60", sommi_htc_res2="240",
    sommi_diamond_wt_bear_level=0.0, sommi_diamond_wt_bull_level=0.0,
)


# --------------------------------------------------------------------------- pieces
def wavetrend(src, chlen: int = 9, avg: int = 12, malen: int = 3):
    """f_wavetrend on the chart timeframe: returns (wt1, wt2, wt_vwap)."""
    ap = f64(src)
    esa = ema(ap, chlen)
    de = ema(np.abs(ap - esa), chlen)
    ci = _div(ap - esa, 0.015 * de)
    wt1 = ema(ci, avg)
    wt2 = sma(wt1, malen)
    return wt1, wt2, wt1 - wt2


def rsi_mfi(open_, high, low, close, period: int = 60, multiplier: float = 150.0, pos_y: float = 2.5):
    """f_rsimfi: sma(((close - open) / (high - low)) * mult, period) - posY."""
    o, h, l, c = f64(open_), f64(high), f64(low), f64(close)
    return sma(_div(c - o, h - l) * multiplier, period) - pos_y


def schaff_tc(src, length: int = 10, fast: int = 23, slow: int = 50, factor: float = 0.5):
    """f_tc (Schaff Trend Cycle). Pine-faithful: before the window is complete
    ``nz(gamma[1])`` makes the cycle start at 0 instead of na."""
    s = f64(src)
    macd_v = ema(s, fast) - ema(s, slow)
    return _stc_k(macd_v, lowest(macd_v, length), highest(macd_v, length), int(length), float(factor))


from numba import njit  # noqa: E402


@njit(cache=True)
def _roll_ext(x, i, n, is_max):
    if i < n - 1:
        return np.nan
    best = x[i]
    if np.isnan(best):
        return np.nan
    for j in range(i - n + 1, i):
        v = x[j]
        if np.isnan(v):
            return np.nan
        if is_max:
            if v > best:
                best = v
        else:
            if v < best:
                best = v
    return best


@njit(cache=True)
def _stc_k(macd_v, lo, hi, length, factor):
    n = len(macd_v)
    gamma = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    eta = np.full(n, np.nan)
    stc = np.full(n, np.nan)
    for i in range(n):
        alpha = lo[i]
        beta = hi[i] - alpha
        if beta > 0:
            g = (macd_v[i] - alpha) / beta * 100.0
        else:
            g = gamma[i - 1] if i > 0 else np.nan
            if np.isnan(g):
                g = 0.0
        gamma[i] = g
        dprev = delta[i - 1] if i > 0 else np.nan
        delta[i] = g if np.isnan(dprev) else dprev + factor * (g - dprev)
        eps = _roll_ext(delta, i, length, False)
        zeta = _roll_ext(delta, i, length, True) - eps
        if zeta > 0:
            e = (delta[i] - eps) / zeta * 100.0
        else:
            e = eta[i - 1] if i > 0 else np.nan
            if np.isnan(e):
                e = 0.0
        eta[i] = e
        sprev = stc[i - 1] if i > 0 else np.nan
        stc[i] = e if np.isnan(sprev) else sprev + factor * (e - sprev)
    return stc


def find_divs(src, high, low, top_limit: float, bot_limit: float, use_limits: bool) -> dict:
    """f_findDivs. Signals on the confirmation bar (fractal centre = t-2)."""
    s = f64(src)
    h, l = f64(high), f64(low)
    s1, s2, s3, s4 = shift(s, 1), shift(s, 2), shift(s, 3), shift(s, 4)
    top = (s4 < s2) & (s3 < s2) & (s2 > s1) & (s2 > s)
    bot = (s4 > s2) & (s3 > s2) & (s2 < s1) & (s2 < s)
    fr = np.where(top, 1, np.where(bot, -1, 0))
    top_ok = (fr > 0) & ((s2 >= top_limit) if use_limits else True)
    bot_ok = (fr < 0) & ((s2 <= bot_limit) if use_limits else True)
    fractal_top = np.where(top_ok, s2, np.nan)
    fractal_bot = np.where(bot_ok, s2, np.nan)
    ctop = _truthy(fractal_top)  # Pine float->bool: na and 0 are false
    cbot = _truthy(fractal_bot)
    h2, l2 = shift(h, 2), shift(l, 2)
    high_prev = shift(valuewhen(ctop, s2, 0), 2)
    high_price = shift(valuewhen(ctop, h2, 0), 2)
    low_prev = shift(valuewhen(cbot, s2, 0), 2)
    low_price = shift(valuewhen(cbot, l2, 0), 2)
    return {
        "fractal_top": fractal_top,
        "fractal_bot": fractal_bot,
        "low_prev": low_prev,
        "bear": ctop & (h2 > high_price) & (s2 < high_prev),
        "bull": cbot & (l2 < low_price) & (s2 > low_prev),
        "bear_hidden": ctop & (h2 < high_price) & (s2 > high_prev),
        "bull_hidden": cbot & (l2 > low_price) & (s2 < low_prev),
    }


def _ha_dir(df: pd.DataFrame) -> np.ndarray:
    ho, _, _, hc = heikinashi(df["open"].to_numpy(), df["high"].to_numpy(),
                              df["low"].to_numpy(), df["close"].to_numpy())
    return hc > ho


def _htf_wt_vwap(chlen, avg, malen):
    def f(df: pd.DataFrame):
        src = hlc3(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy())
        return wavetrend(src, chlen, avg, malen)[2]
    return f


# --------------------------------------------------------------------------- main
def cipher_b(open_, high, low, close, time=None, base_tf=None, **params) -> dict:
    """Compute all Cipher B series. ``time``: open times (DatetimeIndex / datetime64 /
    epoch-ms ints) needed for the HTF parts (Sommi flag/diamond); without it those are
    False / NaN. Unknown params raise."""
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown Cipher B params: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    o, h, l, c = f64(open_), f64(high), f64(low), f64(close)
    n = len(c)
    ap = hlc3(h, l, c)

    # RSI, RSI+MFI, WaveTrend, Stoch RSI, STC
    r = rsi(c, p["rsi_len"])
    rsimfi = rsi_mfi(o, h, l, c, p["rsi_mfi_period"], p["rsi_mfi_multiplier"], p["rsi_mfi_pos_y"])
    wt1, wt2, wt_vwap = wavetrend(ap, p["wt_channel_len"], p["wt_average_len"], p["wt_ma_len"])
    wt_oversold = wt2 <= p["os_level"]
    wt_overbought = wt2 >= p["ob_level"]
    wt_cross = cross(wt1, wt2)
    wt_cross_up = (wt2 - wt1) <= 0
    wt_cross_down = (wt2 - wt1) >= 0
    kk, d1 = stoch_rsi_kd(c, p["stoch_rsi_len"], p["stoch_len"], p["stoch_k_smooth"],
                          p["stoch_d_smooth"], p["stoch_use_log"])
    stoch_k = (kk + d1) / 2.0 if p["stoch_avg"] else kk
    stc = schaff_tc(c, p["tc_length"], p["tc_fast_length"], p["tc_slow_length"], p["tc_factor"])

    # Divergences
    wtd = find_divs(wt2, h, l, p["wt_div_ob_level"], p["wt_div_os_level"], True)
    wtd_add = find_divs(wt2, h, l, p["wt_div_ob_level_add"], p["wt_div_os_level_add"], True)
    wtd_nl = find_divs(wt2, h, l, 0.0, 0.0, False)
    rsd = find_divs(r, h, l, p["rsi_div_ob_level"], p["rsi_div_os_level"], True)
    rsd_nl = find_divs(r, h, l, 0.0, 0.0, False)
    std = find_divs(stoch_k, h, l, 0.0, 0.0, False)
    nl = p["show_hidden_div_nl"]
    wt_bear_hidden = wtd_nl["bear_hidden"] if nl else wtd["bear_hidden"]
    wt_bull_hidden = wtd_nl["bull_hidden"] if nl else wtd["bull_hidden"]
    rsi_bear_hidden = rsd_nl["bear_hidden"] if nl else rsd["bear_hidden"]
    rsi_bull_hidden = rsd_nl["bull_hidden"] if nl else rsd["bull_hidden"]

    # Circles
    buy = wt_cross & wt_cross_up & wt_oversold
    sell = wt_cross & wt_cross_down & wt_overbought
    # buySignalDiv = (wtShowDiv and wtBullDiv) or (wtShowDiv and wtBullDiv_add) or ...
    # (wtDivOBLevel_addshow only affects colours in the original, not the signal)
    show_wt = p["wt_show_div"]
    show_st, show_rsi = p["stoch_show_div"], p["rsi_show_div"]
    F = np.zeros(n, dtype=bool)
    buy_div = ((wtd["bull"] if show_wt else F) | (wtd_add["bull"] if show_wt else F)
               | (std["bull"] if show_st else F) | (rsd["bull"] if show_rsi else F))
    sell_div = ((wtd["bear"] if show_wt else F) | (wtd_add["bear"] if show_wt else F)
                | (std["bear"] if show_st else F) | (rsd["bear"] if show_rsi else F))

    # Gold buy
    last_rsi = shift(valuewhen(_truthy(wtd["fractal_bot"]), shift(r, 2), 0), 2)
    wt_low_prev = wtd["low_prev"]
    gold_buy = (((wtd["bull"] if show_wt else F) | (rsd["bull"] if show_rsi else F))
                & (wt_low_prev <= p["os_level3"]) & (wt2 > p["os_level3"])
                & ((wt_low_prev - wt2) <= -5) & (last_rsi < 30))

    # Higher timeframe parts
    if time is not None:
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                          index=pd.to_datetime(_to_ms(time), unit="ms", utc=True))
        hvwap = request_security_htf(
            df, p["sommi_vwap_tf"],
            _htf_wt_vwap(p["wt_channel_len"], p["wt_average_len"], p["wt_ma_len"]), base_tf)
        ha1 = request_security_htf(df, p["sommi_htc_res"], _ha_dir, base_tf)
        ha2 = request_security_htf(df, p["sommi_htc_res2"], _ha_dir, base_tf)
        # HA direction before the first closed HTF bar is unknown -> neither bull nor bear
        known1 = request_security_htf(df, p["sommi_htc_res"], lambda d: np.ones(len(d), bool), base_tf)
        known2 = request_security_htf(df, p["sommi_htc_res2"], lambda d: np.ones(len(d), bool), base_tf)
    else:
        hvwap = np.full(n, np.nan)
        ha1 = ha2 = known1 = known2 = F

    flag_bear = ((rsimfi < p["sommi_rsimfi_bear_level"]) & (wt2 > p["sommi_flag_wt_bear_level"])
                 & wt_cross & wt_cross_down & (hvwap < p["sommi_vwap_bear_level"]))
    flag_bull = ((rsimfi > p["sommi_rsimfi_bull_level"]) & (wt2 < p["sommi_flag_wt_bull_level"])
                 & wt_cross & wt_cross_up & (hvwap > p["sommi_vwap_bull_level"]))
    dia_bear = ((wt2 >= p["sommi_diamond_wt_bear_level"]) & wt_cross & wt_cross_down
                & known1 & known2 & ~ha1 & ~ha2)
    dia_bull = ((wt2 <= p["sommi_diamond_wt_bull_level"]) & wt_cross & wt_cross_up & ha1 & ha2)

    stc_up = crossover(stc, 25.0)
    stc_dn = crossunder(stc, 75.0)
    return {
        # series
        "wt1": wt1, "wt2": wt2, "wt_vwap": wt_vwap, "rsi": r, "rsi_mfi": rsimfi,
        "stoch_k": stoch_k, "stoch_d": d1, "stc": stc, "htf_wt_vwap": hvwap,
        "wt_low_prev": wt_low_prev, "last_rsi": last_rsi,
        # WT states / crosses
        "wt_oversold": wt_oversold, "wt_overbought": wt_overbought,
        # Pine's wtCrossUp / wtCrossDown are STATES (wt2 - wt1 <= 0 / >= 0), exposed raw as
        # wt_up_state / wt_down_state; wt_cross_up / wt_cross_down = wtCross and the state.
        "wt_cross": wt_cross, "wt_up_state": wt_cross_up, "wt_down_state": wt_cross_down,
        "wt_cross_up": wt_cross & wt_cross_up,
        "wt_cross_down": wt_cross & wt_cross_down,
        "wt_above_zero": wt2 > 0, "wt_below_zero": wt2 < 0,
        # circles
        "buy": buy, "sell": sell, "gold_buy": gold_buy,
        "bull_div": buy_div, "bear_div": sell_div,
        # divergences (confirmation bar)
        "wt_bull_div": wtd["bull"], "wt_bear_div": wtd["bear"],
        "wt_bull_div_add": wtd_add["bull"], "wt_bear_div_add": wtd_add["bear"],
        "wt_bull_div_hidden": wt_bull_hidden, "wt_bear_div_hidden": wt_bear_hidden,
        "rsi_bull_div": rsd["bull"], "rsi_bear_div": rsd["bear"],
        "rsi_bull_div_hidden": rsi_bull_hidden, "rsi_bear_div_hidden": rsi_bear_hidden,
        "stoch_bull_div": std["bull"], "stoch_bear_div": std["bear"],
        "stoch_bull_div_hidden": std["bull_hidden"], "stoch_bear_div_hidden": std["bear_hidden"],
        # areas / oscillators
        "mfi_green": rsimfi > 0, "mfi_red": rsimfi < 0,
        "stoch_k_cross_up_d": crossover(stoch_k, d1), "stoch_k_cross_down_d": crossunder(stoch_k, d1),
        "stc_cross_up_25": stc_up, "stc_cross_down_75": stc_dn,
        # Sommi
        "sommi_bull_flag": flag_bull, "sommi_bear_flag": flag_bear,
        "sommi_bull_diamond": dia_bull, "sommi_bear_diamond": dia_bear,
    }


def _to_ms(time) -> np.ndarray:
    from .base import times_ms
    return times_ms(time)
