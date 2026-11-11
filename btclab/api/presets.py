"""Named strategy presets (DSL specs)."""
from __future__ import annotations

import copy

PRESETS: dict[str, dict] = {
    "brayvb_v2": {
        "name": "brayvb_v2", "tf": "4h",
        "indicators": {"gc": {}, "stochrsi": {}},
        "long": ["gc.filter_rising", "gc.close_above_hband",
                 ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]],
        "exit_long": ["gc.cross_below_hband"],
        "risk": {}, "costs": "BINANCE_PERP_TAKER",
    },
}


# Tuned variants from research/brayvb_tune.py (all-in-sample optimum; NOT validated, see research/reports/brayvb_tune.md)
PRESETS["brayvb_v2_tuned_1h"] = {"name": "brayvb_v2_tuned_1h", "tf": "1h", "indicators": {"gc": {"poles": 6, "period": 178, "mult": 2.3, "mode_lag": False, "mode_fast": False}, "stochrsi": {"rsi_len": 7, "stoch_len": 17, "smooth_k": 5, "smooth_d": 5, "ob": 80.0, "os": 15.0, "use_log": False}}, "long": ["gc.filter_rising", "gc.close_above_hband", ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]], "exit_long": ["gc.cross_below_hband"], "risk": {"sl_atr": 1.5}, "costs": "BINANCE_PERP_TAKER"}
PRESETS["brayvb_v2_tuned_1h_ha"] = {"name": "brayvb_v2_tuned_1h_ha", "tf": "1h", "indicators": {"gc": {"poles": 6, "period": 90, "mult": 2.4, "mode_lag": False, "mode_fast": True}, "stochrsi": {"rsi_len": 10, "stoch_len": 7, "smooth_k": 2, "smooth_d": 5, "ob": 85.0, "os": 10.0, "use_log": True}}, "long": ["gc.filter_rising", "gc.close_above_hband", ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]], "exit_long": ["gc.cross_below_hband"], "risk": {"sl_atr": 2.0, "trail_atr": 4.0}, "costs": "BINANCE_PERP_TAKER", "source": "ha"}
PRESETS["brayvb_v2_tuned_4h"] = {"name": "brayvb_v2_tuned_4h", "tf": "4h", "indicators": {"gc": {"poles": 4, "period": 102, "mult": 1.7, "mode_lag": False, "mode_fast": True}, "stochrsi": {"rsi_len": 18, "stoch_len": 15, "smooth_k": 2, "smooth_d": 2, "ob": 80.0, "os": 10.0, "use_log": True}}, "long": ["gc.filter_rising", "gc.close_above_hband", ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]], "exit_long": ["gc.cross_below_hband"], "risk": {"sl_atr": 2.0, "trail_atr": 3.0}, "costs": "BINANCE_PERP_TAKER"}
PRESETS["brayvb_v2_tuned_4h_ha"] = {"name": "brayvb_v2_tuned_4h_ha", "tf": "4h", "indicators": {"gc": {"poles": 3, "period": 74, "mult": 1.9, "mode_lag": False, "mode_fast": True}, "stochrsi": {"rsi_len": 20, "stoch_len": 12, "smooth_k": 2, "smooth_d": 3, "ob": 90.0, "os": 10.0, "use_log": True}}, "long": ["gc.filter_rising", "gc.close_above_hband", ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]], "exit_long": ["gc.cross_below_hband"], "risk": {"sl_atr": 3.0}, "costs": "BINANCE_PERP_TAKER", "source": "ha"}
PRESETS["brayvb_v2_tuned_1d"] = {"name": "brayvb_v2_tuned_1d", "tf": "1d", "indicators": {"gc": {"poles": 6, "period": 176, "mult": 1.4, "mode_lag": False, "mode_fast": True}, "stochrsi": {"rsi_len": 19, "stoch_len": 10, "smooth_k": 2, "smooth_d": 2, "ob": 80.0, "os": 10.0, "use_log": False}}, "long": ["gc.filter_rising", "gc.close_above_hband", ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]], "exit_long": ["gc.cross_below_hband"], "risk": {"sl_atr": 3.0, "trail_atr": 1.0}, "costs": "BINANCE_PERP_TAKER"}
PRESETS["brayvb_v2_tuned_1d_ha"] = {"name": "brayvb_v2_tuned_1d_ha", "tf": "1d", "indicators": {"gc": {"poles": 4, "period": 132, "mult": 1.8, "mode_lag": False, "mode_fast": False}, "stochrsi": {"rsi_len": 15, "stoch_len": 9, "smooth_k": 2, "smooth_d": 5, "ob": 90.0, "os": 10.0, "use_log": False}}, "long": ["gc.filter_rising", "gc.close_above_hband", ["stochrsi.k_ge_ob", "stochrsi.k_le_os"]], "exit_long": ["gc.cross_below_hband"], "risk": {"sl_atr": 3.0, "trail_atr": 1.0}, "costs": "BINANCE_PERP_TAKER", "source": "ha"}


# Amendment 8 long/short presets (research/a8). NONE passed V1+V2: marked 'nao validada' (top 3 by V1 EW Sharpe). See research/reports/a8_validation.md
PRESETS['ls_ema_hull_1d_ha'] = {"name": "ls_ema_hull_1d_ha", "tf": "1d", "indicators": {"hull": {"length": 15}, "macross": {"kind": "sma", "slow": 100, "fast": 20}}, "costs": "BINANCE_PERP_TAKER", "risk": {}, "long": ["hull.close_cross_up", "macross.fast_above_slow"], "exit_long": ["hull.falling"], "short": ["hull.close_cross_down", "macross.fast_below_slow"], "exit_short": ["hull.rising"], "source": "ha", "note": "nao validada (A8: reprovada em V1/V2)"}
PRESETS['ls_ema_gc_1d'] = {"name": "ls_ema_gc_1d", "tf": "1d", "indicators": {"gc": {"poles": 2, "period": 100, "mult": 2.3, "mode_lag": True, "mode_fast": True}, "macross": {"kind": "ema", "slow": 100, "fast": 20}}, "costs": "BINANCE_PERP_TAKER", "risk": {}, "long": ["gc.cross_above_filt", "macross.close_above_slow"], "exit_long": ["gc.close_below_filt"], "short": ["gc.cross_below_filt", "macross.close_below_slow"], "exit_short": ["gc.close_above_filt"], "note": "nao validada (A8: reprovada em V1/V2)"}
PRESETS['ls_st_ssl_4h_ha'] = {"name": "ls_st_ssl_4h_ha", "tf": "4h", "indicators": {"ssl": {"length": 44}, "supertrend": {"atr_len": 23, "factor": 3.25}}, "costs": "BINANCE_PERP_TAKER", "risk": {}, "long": ["ssl.flip_up", "supertrend.up"], "exit_long": ["ssl.bear"], "short": ["ssl.flip_down", "supertrend.down"], "exit_short": ["ssl.bull"], "source": "ha", "note": "nao validada (A8: reprovada em V1/V2)"}

PRESETS['ls_adx_gc_1d'] = {'name': 'ls_adx_gc_1d', 'tf': '1d', 'indicators': {'gc': {'poles': 5, 'period': 96, 'mult': 2.3, 'mode_lag': False, 'mode_fast': True}, 'dmi': {'di_len': 9, 'adx_len': 30, 'adx_thr': 17.0}}, 'costs': 'BINANCE_PERP_TAKER', 'risk': {}, 'long': ['gc.cross_above_filt', 'dmi.bull_strong'], 'exit_long': ['gc.close_below_filt'], 'short': ['gc.cross_below_filt', 'dmi.bear_strong'], 'exit_short': ['gc.close_above_filt'], 'note': 'nao validada (A8 a8c9: passou V2, reprovou V1 por Sharpe mediano 0.44)'}


def get_preset(name: str) -> dict:
    return copy.deepcopy(PRESETS[name])
