"""Pine Script v6 emitter for DSL strategy specs.

``emit_strategy(spec, name)`` renders a ``strategy()`` script whose conditions mirror the
DSL (top-level AND, nested list OR, nested-in-OR AND, ``not:`` prefix). Each indicator in
``spec["indicators"]`` gets a declaration template (_DECLS) and each primitive a boolean
expression template (_PRIMS); ``{p}`` is the indicator prefix. Unknown primitives raise
NotImplementedError listing all of them.
"""
from __future__ import annotations

from pathlib import Path

from btclab.engine import costs as _costs
from btclab.indicators.registry import REGISTRY
from btclab.signals.dsl import SIDES, validate_spec

_GC_FUNCS = (Path(__file__).with_name("_gc_funcs.txt")).read_text(encoding="utf8")


def _b(v) -> str:
    return "true" if v is True else "false" if v is False else str(v)


# ------------------------------------------------------------------ declarations
def _decl_gc(p, q):
    return [
        _GC_FUNCS.rstrip(),
        f"{p}_beta = (1 - math.cos(4*math.asin(1)/{q['period']})) / (math.pow(1.414, 2/{q['poles']}) - 1)",
        f"{p}_alpha = -{p}_beta + math.sqrt(math.pow({p}_beta, 2) + 2*{p}_beta)",
        f"{p}_lag = int(({q['period']} - 1)/(2*{q['poles']}))",
        f"{p}_src = hlc3",
        f"{p}_tr = ta.tr(true)",
        f"{p}_srcdata = {_b(q['mode_lag'])} ? {p}_src + ({p}_src - {p}_src[{p}_lag]) : {p}_src",
        f"{p}_trdata = {_b(q['mode_lag'])} ? {p}_tr + ({p}_tr - {p}_tr[{p}_lag]) : {p}_tr",
        f"[{p}_filtn, {p}_filt1] = f_pole({p}_alpha, {p}_srcdata, {q['poles']})",
        f"[{p}_filtntr, {p}_filt1tr] = f_pole({p}_alpha, {p}_trdata, {q['poles']})",
        f"{p}_filt = {_b(q['mode_fast'])} ? ({p}_filtn + {p}_filt1)/2 : {p}_filtn",
        f"{p}_filttr = {_b(q['mode_fast'])} ? ({p}_filtntr + {p}_filt1tr)/2 : {p}_filtntr",
        f"{p}_hband = {p}_filt + {p}_filttr*{q['mult']}",
        f"{p}_lband = {p}_filt - {p}_filttr*{q['mult']}",
    ]


def _decl_stochrsi(p, q):
    if q["use_log"]:
        raise NotImplementedError("stochrsi use_log=True has no Pine template")
    return [
        f"{p}_rsi = ta.rsi(close, {q['rsi_len']})",
        f"{p}_k = ta.sma(ta.stoch({p}_rsi, {p}_rsi, {p}_rsi, {q['stoch_len']}), {q['smooth_k']})",
        f"{p}_d = ta.sma({p}_k, {q['smooth_d']})",
    ]


def _decl_macd(p, q):
    return [f"[{p}_m, {p}_s, {p}_h] = ta.macd(close, {q['fast']}, {q['slow']}, {q['signal']})"]


def _decl_macross(p, q):
    fn = {"ema": "ta.ema", "sma": "ta.sma", "rma": "ta.rma", "wma": "ta.wma"}[q["kind"]]
    return [f"{p}_fast = {fn}(close, {q['fast']})", f"{p}_slow = {fn}(close, {q['slow']})"]


def _decl_bb(p, q):
    return [f"[{p}_basis, {p}_upper, {p}_lower] = ta.bb(close, {q['length']}, {q['mult']})"]


def _decl_donchian(p, q):
    return [f"{p}_up = ta.highest(high, {q['length']})[1]", f"{p}_dn = ta.lowest(low, {q['length']})[1]"]


def _decl_supertrend(p, q):
    return [f"[{p}_st, {p}_dir] = ta.supertrend({q['factor']}, {q['atr_len']})"]


_DECLS = {
    "gc": _decl_gc, "stochrsi": _decl_stochrsi, "macd": _decl_macd, "macross": _decl_macross,
    "bb": _decl_bb, "donchian": _decl_donchian, "supertrend": _decl_supertrend,
    "cci": lambda p, q: [f"{p}_v = ta.cci(hlc3, {q['length']})"],
    "willr": lambda p, q: [f"{p}_v = ta.wpr({q['length']})"],
    "mfi": lambda p, q: [f"{p}_v = ta.mfi(hlc3, {q['length']})"],
}


# ------------------------------------------------------------------ primitive expressions
_PRIMS = {
    "gc.filter_rising": "{p}_filt > {p}_filt[1]",
    "gc.filter_falling": "{p}_filt < {p}_filt[1]",
    "gc.close_above_hband": "close > {p}_hband",
    "gc.close_below_lband": "close < {p}_lband",
    "gc.close_above_filt": "close > {p}_filt",
    "gc.close_below_filt": "close < {p}_filt",
    "gc.cross_above_hband": "ta.crossover(close, {p}_hband)",
    "gc.cross_below_hband": "ta.crossunder(close, {p}_hband)",
    "gc.cross_above_filt": "ta.crossover(close, {p}_filt)",
    "gc.cross_below_filt": "ta.crossunder(close, {p}_filt)",
    "gc.cross_above_lband": "ta.crossover(close, {p}_lband)",
    "gc.cross_below_lband": "ta.crossunder(close, {p}_lband)",
    "stochrsi.k_cross_up_d": "ta.crossover({p}_k, {p}_d)",
    "stochrsi.k_cross_down_d": "ta.crossunder({p}_k, {p}_d)",
    "stochrsi.k_above_ob": "{p}_k > {ob}",
    "stochrsi.k_below_os": "{p}_k < {os}",
    "stochrsi.k_ge_ob": "{p}_k >= {ob}",
    "stochrsi.k_le_os": "{p}_k <= {os}",
    "stochrsi.k_extreme": "({p}_k >= {ob} or {p}_k <= {os})",
    "stochrsi.k_cross_up_os": "ta.crossover({p}_k, {os})",
    "stochrsi.k_cross_down_ob": "ta.crossunder({p}_k, {ob})",
    "stochrsi.k_above_50": "{p}_k > 50",
    "macd.cross_up": "ta.crossover({p}_m, {p}_s)",
    "macd.cross_down": "ta.crossunder({p}_m, {p}_s)",
    "macd.cross_up_below_zero": "(ta.crossover({p}_m, {p}_s) and {p}_m < 0)",
    "macd.cross_down_above_zero": "(ta.crossunder({p}_m, {p}_s) and {p}_m > 0)",
    "macd.above_zero": "{p}_m > 0",
    "macd.below_zero": "{p}_m < 0",
    "macd.hist_rising": "{p}_h > {p}_h[1]",
    "macd.zero_cross_up": "ta.crossover({p}_m, 0)",
    "macd.zero_cross_down": "ta.crossunder({p}_m, 0)",
    "macross.fast_above_slow": "{p}_fast > {p}_slow",
    "macross.fast_below_slow": "{p}_fast < {p}_slow",
    "macross.cross_up": "ta.crossover({p}_fast, {p}_slow)",
    "macross.cross_down": "ta.crossunder({p}_fast, {p}_slow)",
    "macross.close_above_slow": "close > {p}_slow",
    "macross.close_below_slow": "close < {p}_slow",
    "macross.close_above_fast": "close > {p}_fast",
    "cci.cross_up_os": "ta.crossover({p}_v, -{level})",
    "cci.cross_down_ob": "ta.crossunder({p}_v, {level})",
    "cci.cross_up_ob": "ta.crossover({p}_v, {level})",
    "cci.cross_down_os": "ta.crossunder({p}_v, -{level})",
    "cci.above_zero": "{p}_v > 0",
    "cci.below_zero": "{p}_v < 0",
    "willr.cross_up_os": "ta.crossover({p}_v, {os})",
    "willr.cross_down_ob": "ta.crossunder({p}_v, {ob})",
    "willr.below_os": "{p}_v < {os}",
    "willr.above_ob": "{p}_v > {ob}",
    "mfi.cross_up_os": "ta.crossover({p}_v, {os})",
    "mfi.cross_down_ob": "ta.crossunder({p}_v, {ob})",
    "mfi.below_os": "{p}_v < {os}",
    "mfi.above_ob": "{p}_v > {ob}",
    "bb.cross_above_upper": "ta.crossover(close, {p}_upper)",
    "bb.cross_below_lower": "ta.crossunder(close, {p}_lower)",
    "bb.cross_above_basis": "ta.crossover(close, {p}_basis)",
    "bb.cross_below_basis": "ta.crossunder(close, {p}_basis)",
    "donchian.breakout_up": "close > {p}_up",
    "donchian.breakout_down": "close < {p}_dn",
    "donchian.close_above_mid": "close > ({p}_up + {p}_dn)/2",
    "donchian.close_below_mid": "close < ({p}_up + {p}_dn)/2",
    "supertrend.up": "{p}_dir < 0",
    "supertrend.down": "{p}_dir > 0",
    "supertrend.flip_up": "({p}_dir < 0 and {p}_dir[1] > 0)",
    "supertrend.flip_down": "({p}_dir > 0 and {p}_dir[1] < 0)",
}


def _split(tok: str):
    neg = tok.startswith("not:")
    return neg, (tok[4:].strip() if neg else tok.strip())


def _walk(node):
    if isinstance(node, str):
        yield _split(node)[1]
    else:
        for n in node:
            yield from _walk(n)


def _render(node, params, is_or=False) -> str:
    if isinstance(node, str):
        neg, prim = _split(node)
        ind = prim.partition(".")[0]
        expr = _PRIMS[prim].format(p=ind, **params[ind])
        return f"not ({expr})" if neg else expr
    parts = [_render(c, params, not is_or) for c in node]
    if len(parts) == 1:
        return parts[0]
    return "(" + (" or " if is_or else " and ").join(parts) + ")"


def _cost_model(name):
    for k, m in _costs.PRESETS.items():
        if str(name).lower() in (k.lower(), m.name.lower()):
            return m
    raise ValueError(f"unknown cost model {name!r}")


def emit_strategy(spec: dict, name: str | None = None) -> str:
    inds = spec.get("indicators", {})
    prims = [pr for s in SIDES for pr in _walk(spec.get(s) or [])]
    missing = sorted({pr for pr in prims
                      if pr not in _PRIMS or pr.partition(".")[0] not in _DECLS})
    if missing:
        raise NotImplementedError(f"no Pine template for primitives: {', '.join(missing)}")
    validate_spec(spec)
    name = name or spec["name"]
    used = {pr.partition(".")[0] for pr in prims}

    params = {n: {**REGISTRY[n].defaults, **(inds[n] or {})} for n in inds}
    cm = _cost_model(spec.get("costs", "BINANCE_PERP_TAKER"))
    risk = {k: v for k, v in (spec.get("risk") or {}).items() if v}
    do_short = bool(spec.get("short")) and cm.allow_short

    out = [
        "//@version=6",
        f"// GENERATED by btclab.pine.emit_strategy from DSL spec '{spec['name']}' (tf {spec['tf']}) - do not edit by hand.",
        f"// Costs preset {cm.name}: fee {cm.fee_rate*100:g}%/side. Engine slippage {cm.slippage_bp:g} bp/side is NOT modelled",
        "// here (set Properties > Slippage in TradingView to compare). Orders fill at the next bar open.",
        f'strategy(title="{name}", overlay=true, initial_capital=100000, '
        "default_qty_type=strategy.percent_of_equity, default_qty_value=100, pyramiding=1, "
        f"commission_type=strategy.commission.percent, commission_value={cm.fee_rate*100:g}, "
        "margin_long=100, margin_short=100, process_orders_on_close=false)",
        "",
    ]
    for n in inds:
        if n in used:
            out += [f"// --- {n} ---"] + _DECLS[n](n, params[n]) + [""]
    if risk:
        out += ["atrv = ta.atr(14)", ""]

    out.append("// --- signals ---")
    for s in SIDES:
        if spec.get(s):
            out.append(f"{s}_sig = {_render(list(spec[s]), params)}")
    out.append("")
    if spec.get("long"):
        out += ["if long_sig", '    strategy.entry("long", strategy.long)']
    if do_short:
        out += ["if short_sig", '    strategy.entry("short", strategy.short)']
    if spec.get("exit_long"):
        out += ["if exit_long_sig", '    strategy.close("long")']
    if spec.get("exit_short"):
        out += ["if exit_short_sig", '    strategy.close("short")']

    if risk:
        out += ["", "// --- risk: ATR(14) taken while flat (signal bar) ---",
                "var float atrE = na",
                "if strategy.position_size == 0",
                "    atrE := atrv"]
        for side, sgn in (("long", 1), ("short", -1)):
            if side == "short" and not do_short:
                continue
            args = []
            if "sl_atr" in risk:
                args.append(f"stop=strategy.position_avg_price - ({sgn})*{risk['sl_atr']}*atrE")
            if "tp_atr" in risk:
                args.append(f"limit=strategy.position_avg_price + ({sgn})*{risk['tp_atr']}*atrE")
            if "trail_atr" in risk:
                args.append(f"trail_points={risk['trail_atr']}*atrE/syminfo.mintick, trail_offset=0")
            if args:
                cond = "strategy.position_size > 0" if side == "long" else "strategy.position_size < 0"
                out += [f"if {cond}", f'    strategy.exit("{side}_x", "{side}", {", ".join(args)})']
        if "time_stop" in risk:
            out += ["if strategy.position_size != 0 and bar_index - strategy.opentrades.entry_bar_index(0) "
                    f">= {int(risk['time_stop'])}",
                    '    strategy.close_all("time_stop")']
    return "\n".join(out) + "\n"
