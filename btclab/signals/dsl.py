"""JSON strategy spec.

{name, tf, indicators:{name:params}, long, exit_long, short, exit_short,
 risk:{sl_atr,tp_atr,trail_atr,time_stop}, costs:"BINANCE_PERP_TAKER"}

Condition lists: the top-level list is AND; a nested list is OR (and a list nested
inside an OR is AND again); a string prefixed "not:" is negated.
Primitives are "<indicator>.<output>" and must exist in the indicator registry.
"""
from __future__ import annotations

import numpy as np

from btclab.indicators.registry import REGISTRY

SIDES = ("long", "exit_long", "short", "exit_short")
RISK_KEYS = ("sl_atr", "tp_atr", "trail_atr", "time_stop")


class SpecError(ValueError):
    pass


def _split(token: str):
    neg = token.startswith("not:")
    prim = token[4:].strip() if neg else token.strip()
    return neg, prim


def _walk(node, fn):
    if isinstance(node, str):
        fn(_split(node)[1])
    elif isinstance(node, (list, tuple)):
        for n in node:
            _walk(n, fn)
    else:
        raise SpecError(f"invalid condition node {node!r}")


def validate_spec(spec: dict) -> None:
    for k in ("name", "tf", "indicators"):
        if k not in spec:
            raise SpecError(f"missing key {k!r}")
    inds = spec["indicators"]
    for n in inds:
        if n not in REGISTRY:
            raise SpecError(f"unknown indicator {n!r}")
        bad = set(inds[n] or {}) - set(REGISTRY[n].defaults)
        if bad:
            raise SpecError(f"unknown params {sorted(bad)} for {n!r}")

    def check(prim):
        name = prim.partition(".")[0]
        if name not in REGISTRY or prim not in REGISTRY[name].primitives:
            raise SpecError(f"unknown primitive {prim!r}")
        if name not in inds:
            raise SpecError(f"primitive {prim!r} uses indicator {name!r} not in 'indicators'")

    for side in SIDES:
        _walk(spec.get(side) or [], check)
    bad = set(spec.get("risk") or {}) - set(RISK_KEYS)
    if bad:
        raise SpecError(f"unknown risk keys {sorted(bad)}")


def eval_cond(node, lookup, n: int, is_or: bool = False) -> np.ndarray:
    """Evaluate a condition tree. ``lookup(prim) -> bool array``."""
    if isinstance(node, str):
        neg, prim = _split(node)
        a = np.asarray(lookup(prim), dtype=bool)
        return ~a if neg else a
    parts = [eval_cond(c, lookup, n, not is_or) for c in node]
    out = np.zeros(n, bool) if is_or else np.ones(n, bool)
    for p in parts:
        out = (out | p) if is_or else (out & p)
    return out


def build_signal(conds, lookup, n: int):
    """Top-level list = AND. Empty/None -> None (no signal)."""
    if not conds:
        return None
    return eval_cond(list(conds), lookup, n, is_or=False)
