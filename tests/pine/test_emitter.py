from pathlib import Path

import pytest

from btclab.api.presets import get_preset
from btclab.pine import emit_strategy

ROOT = Path(__file__).resolve().parents[2]


def test_brayvb_v2_matches_reference_and_writes_file():
    out = emit_strategy(get_preset("brayvb_v2"), "brayvb v2 (generated)")
    gen = ROOT / "research/pine/generated"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "brayvb_v2.pine").write_text(out, encoding="utf8")
    for s in ["//@version=6", "GENERATED", "process_orders_on_close=false", "commission_value=0.05",
              "f_filt9x(_a, _s, _i)", "ta.rsi(close, 14)",
              "ta.sma(ta.stoch(stochrsi_rsi, stochrsi_rsi, stochrsi_rsi, 14), 3)",
              "gc_hband = gc_filt + gc_filttr*1.414", "src = hlc3" if False else "gc_src = hlc3",
              'strategy.entry("long", strategy.long)', 'strategy.close("long")']:
        assert s in out, s
    assert ("long_sig = (gc_filt > gc_filt[1] and close > gc_hband and "
            "(stochrsi_k >= 80.0 or stochrsi_k <= 20.0))") in out
    assert "exit_long_sig = ta.crossunder(close, gc_hband)" in out
    assert "strategy.short" not in out


def test_nesting_and_not():
    spec = {"name": "t", "tf": "1h", "indicators": {"macross": {}, "macd": {}, "cci": {}},
            "long": ["macross.cross_up", [["macd.above_zero", "not:cci.above_zero"], "macd.cross_up"]],
            "short": ["macross.cross_down"], "exit_long": ["macross.cross_down"],
            "risk": {"sl_atr": 2, "tp_atr": 3}}
    out = emit_strategy(spec, "t")
    assert ("long_sig = (ta.crossover(macross_fast, macross_slow) and "
            "((macd_m > 0 and not (cci_v > 0)) or ta.crossover(macd_m, macd_s)))") in out
    assert 'strategy.entry("short", strategy.short)' in out
    assert "stop=strategy.position_avg_price - (1)*2*atrE" in out
    assert "limit=strategy.position_avg_price + (-1)*3*atrE" in out


def test_unknown_primitive_raises():
    spec = {"name": "t", "tf": "1h", "indicators": {"qqe": {}, "ob": {}},
            "long": ["qqe.bull", "ob.bull_ob"]}
    with pytest.raises(NotImplementedError) as e:
        emit_strategy(spec, "t")
    assert "qqe.bull" in str(e.value) and "ob.bull_ob" in str(e.value)
