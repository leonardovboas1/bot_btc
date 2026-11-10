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


def get_preset(name: str) -> dict:
    return copy.deepcopy(PRESETS[name])
