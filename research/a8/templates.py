"""Amendment 8 templates T1/T2/T3 built from existing registry primitives.

MAPPING (named indicator -> registry primitive actually used)
- EMA/SMA 50/100/200 price-above  -> macross (kind ema|sma, slow in 50/100/200): close_above_slow / close_below_slow
  (no slope primitive exists: the 'slope' variant is mapped to fast_above_slow / fast_below_slow, fast in 10/20)
- Supertrend filter               -> supertrend.up / supertrend.down
- Ichimoku cloud filter           -> ichimoku.above_cloud / below_cloud
- ADX>theta filter                -> dmi.bull_strong / bear_strong (DI+ > DI- and ADX > thr)
- Gaussian Channel filter         -> gc.filter_rising / filter_falling
- MACD trigger                    -> macd.cross_up/cross_down ("cross") or macd.hist_rising / not:hist_rising ("hist")
- RSI cross of levels             -> NO rsi indicator in registry; closest = mfi (volume-weighted RSI): cross_up_os / cross_down_ob
- Stoch RSI K/D cross in zones    -> stochrsi.k_cross_up_d / k_cross_down_d ("kd") or bull_cross_os / bear_cross_ob ("zone")
- Cipher B WT cross               -> cipherb.wt_cross_up / wt_cross_down
- QQE cross                       -> qqe.cross_up / cross_down
- Bollinger re-entry              -> no re-entry primitive; closest = bb.cross_above_basis/cross_below_basis ("basis")
                                     or bb.cross_above_upper/cross_below_lower ("breakout")
- T2 SAR indicators: supertrend (flip_up/flip_down, state up/down), utbot (buy/sell, long_pos/short_pos),
  psar (flip_long/flip_short, long/short), ssl (flip_up/flip_down, bull/bear), rangefilter (buy/sell, bull/bear),
  gc (cross_above_filt/cross_below_filt, close_above_filt/close_below_filt), hull (close_cross_up/down, rising/falling),
  donchian (breakout_up/down; exit when close crosses mid: close_below_mid / close_above_mid).
"""
from __future__ import annotations

from btclab.indicators.registry import REGISTRY

SOURCES = ("real", "ha")
FILTERS = ("ema", "st", "ich", "adx", "gc")
TRIGGERS = ("macd", "mfi", "stochrsi", "cipherb", "qqe", "bb")
SAR = ("supertrend", "utbot", "psar", "ssl", "rangefilter", "gc", "hull", "donchian")
EXIT_STYLES = ("opp", "flip", "opp+flip", "opp+atr", "flip+atr")
SAR_EXITS = ("sar", "sar+atr")

_SAR = {  # ind: (entry_long, entry_short, exit_long (=bear state), exit_short (=bull state))
    "supertrend": ("flip_up", "flip_down", "down", "up"),
    "utbot": ("buy", "sell", "short_pos", "long_pos"),
    "psar": ("flip_long", "flip_short", "short", "long"),
    "ssl": ("flip_up", "flip_down", "bear", "bull"),
    "rangefilter": ("buy", "sell", "bear", "bull"),
    "gc": ("cross_above_filt", "cross_below_filt", "close_below_filt", "close_above_filt"),
    "hull": ("close_cross_up", "close_cross_down", "falling", "rising"),
    "donchian": ("breakout_up", "breakout_down", "close_below_mid", "close_above_mid"),
}
_FILT = {"st": ("supertrend", "supertrend.up", "supertrend.down"),
         "ich": ("ichimoku", "ichimoku.above_cloud", "ichimoku.below_cloud"),
         "adx": ("dmi", "dmi.bull_strong", "dmi.bear_strong"),
         "gc": ("gc", "gc.filter_rising", "gc.filter_falling")}


def filter_prims(f, p):
    """-> (indicator, indicator params, bull prim, bear prim)"""
    if f == "ema":
        ip = dict(kind=p["kind"], slow=p["slow"], fast=p["fast"])
        if p["mode"] == "price":
            return "macross", ip, "macross.close_above_slow", "macross.close_below_slow"
        return "macross", ip, "macross.fast_above_slow", "macross.fast_below_slow"
    ind, bull, bear = _FILT[f]
    return ind, dict(p), bull, bear


def trigger_prims(t, p):
    """-> (indicator, indicator params, long prim, short prim)"""
    mode = p.get("mode")
    ip = {k: v for k, v in p.items() if k != "mode"}
    if t == "macd":
        return ("macd", ip, "macd.cross_up", "macd.cross_down") if mode == "cross" else \
            ("macd", ip, "macd.hist_rising", "not:macd.hist_rising")
    if t == "mfi":
        return "mfi", ip, "mfi.cross_up_os", "mfi.cross_down_ob"
    if t == "stochrsi":
        return ("stochrsi", ip, "stochrsi.k_cross_up_d", "stochrsi.k_cross_down_d") if mode == "kd" else \
            ("stochrsi", ip, "stochrsi.bull_cross_os", "stochrsi.bear_cross_ob")
    if t == "cipherb":
        return "cipherb", ip, "cipherb.wt_cross_up", "cipherb.wt_cross_down"
    if t == "qqe":
        return "qqe", ip, "qqe.cross_up", "qqe.cross_down"
    if t == "bb":
        return ("bb", ip, "bb.cross_above_basis", "bb.cross_below_basis") if mode == "basis" else \
            ("bb", ip, "bb.cross_above_upper", "bb.cross_below_lower")
    raise ValueError(t)


def _risk(style, p):
    if style.endswith("+atr"):
        r = {"sl_atr": float(p["sl_atr"])}
        if p.get("trail_atr", 0):
            r["trail_atr"] = float(p["trail_atr"])
        return r
    return {}


def build_t1(filter_, trigger, params, tf, source="real", sides="both", name=None):
    """params = {"f": filter params, "t": trigger params (+mode), "exit": style, "risk": {sl_atr, trail_atr}}.
    sides: both = T1, long / short = T3 restrictions."""
    fi, fp, fbull, fbear = filter_prims(filter_, params["f"])
    ti, tp, tl, ts = trigger_prims(trigger, params["t"])
    style = params["exit"]
    long, short = [fbull, tl], [fbear, ts]
    if style in ("opp", "opp+atr"):
        xl, xs = [ts], [tl]
    elif style in ("flip", "flip+atr"):
        xl, xs = [fbear], [fbull]
    else:  # opp+flip (nested list = OR)
        xl, xs = [[ts, fbear]], [[tl, fbull]]
    spec = dict(name=name or f"t1_{filter_}_{trigger}", tf=tf, indicators={fi: fp, ti: tp},
                costs="BINANCE_PERP_TAKER", risk=_risk(style, params.get("risk", {})))
    if sides in ("both", "long"):
        spec["long"], spec["exit_long"] = long, xl
    if sides in ("both", "short"):
        spec["short"], spec["exit_short"] = short, xs
    if source != "real":
        spec["source"] = source
    return spec


def build_t2(ind, params, tf, source="real", filter_=None, name=None, sides="both"):
    """Stop-and-reverse on one indicator, optional filter."""
    el, es, xl, xs = (f"{ind}.{x}" for x in _SAR[ind])
    inds = {ind: dict(params["s"])}
    long, short = [el], [es]
    if filter_:
        fi, fp, fbull, fbear = filter_prims(filter_, params["f"])
        inds[fi] = fp
        long.append(fbull)
        short.append(fbear)
    spec = dict(name=name or f"t2_{ind}", tf=tf, indicators=inds, costs="BINANCE_PERP_TAKER",
                risk=_risk(params["exit"], params.get("risk", {})))
    if sides in ("both", "long"):
        spec["long"], spec["exit_long"] = long, [xl]
    if sides in ("both", "short"):
        spec["short"], spec["exit_short"] = short, [xs]
    if source != "real":
        spec["source"] = source
    return spec


# ---------------------------------------------------------------- Optuna sampling
def _suggest(trial, nm, sp):
    if sp["type"] == "int":
        return trial.suggest_int(nm, sp["low"], sp["high"], step=sp.get("step", 1))
    if sp["type"] == "float":
        return round(trial.suggest_float(nm, sp["low"], sp["high"], step=sp["step"]), 6)
    return trial.suggest_categorical(nm, sp["choices"])


def sample_ind(trial, ind, prefix):
    import optuna
    p = {k: _suggest(trial, f"{prefix}.{k}", sp) for k, sp in REGISTRY[ind].space.items()}
    if not REGISTRY[ind].constraint({**REGISTRY[ind].defaults, **p}):
        raise optuna.TrialPruned()
    return p


def sample_filter(trial, f):
    if f == "ema":
        return dict(kind=trial.suggest_categorical("f.kind", ["ema", "sma"]),
                    slow=trial.suggest_categorical("f.slow", [50, 100, 200]),
                    fast=trial.suggest_categorical("f.fast", [10, 20]),
                    mode=trial.suggest_categorical("f.mode", ["price", "fastslow"]))
    return sample_ind(trial, _FILT[f][0], "f")


def sample_trigger(trial, t):
    p = sample_ind(trial, t, "t")
    modes = {"macd": ["cross", "hist"], "stochrsi": ["kd", "zone"], "bb": ["basis", "breakout"]}
    if t in modes:
        p["mode"] = trial.suggest_categorical("t.mode", modes[t])
    return p


def sample_risk(trial, style):
    if style.endswith("+atr"):
        return dict(sl_atr=trial.suggest_float("r.sl_atr", 1.5, 5.0, step=0.5),
                    trail_atr=trial.suggest_categorical("r.trail_atr", [0.0, 2.0, 3.0, 4.0, 6.0]))
    return {}


def sample(trial, combo):
    if combo["template"] == "T1":
        style = trial.suggest_categorical("exit", list(EXIT_STYLES))
        return dict(f=sample_filter(trial, combo["filter"]), t=sample_trigger(trial, combo["trigger"]),
                    exit=style, risk=sample_risk(trial, style))
    style = trial.suggest_categorical("exit", list(SAR_EXITS))
    p = dict(s=sample_ind(trial, combo["trigger"], "s"), exit=style, risk=sample_risk(trial, style))
    if combo["filter"]:
        p["f"] = sample_filter(trial, combo["filter"])
    return p


def combos(tfs, sources=SOURCES):
    out = []
    for tf in tfs:
        for src in sources:
            for f in FILTERS:
                for t in TRIGGERS:
                    out.append(dict(template="T1", filter=f, trigger=t, tf=tf, source=src))
            for s in SAR:
                for f in (None,) + FILTERS:
                    if f == {"supertrend": "st", "gc": "gc"}.get(s):
                        continue   # an indicator name cannot carry two parameter sets
                    out.append(dict(template="T2", filter=f, trigger=s, tf=tf, source=src))
    return out


def build(combo, params, sides="both", name=None):
    if combo["template"] == "T1":
        return build_t1(combo["filter"], combo["trigger"], params, combo["tf"], combo["source"], sides, name)
    return build_t2(combo["trigger"], params, combo["tf"], combo["source"], combo["filter"], name, sides)
