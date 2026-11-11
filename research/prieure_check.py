"""Independent check of the 'Prieuré Strategy' configs quoted from a third-party optimiser.

Same logic as brayvb_v2 (GC rising + close above hband + Stoch RSI extreme; exit close crossing below hband).
For each config: in-sample full-period stats (what the other platform shows), walk-forward OOS on the
8 design coins, and the post-cutoff window (data seen after the A6 holdout was spent).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))

from btclab.api.presets import get_preset  # noqa: E402
from btclab.engine.metrics import daily_returns  # noqa: E402
from btclab.signals.combiner import run_spec  # noqa: E402
from xasset_brayvb import make_ctx, stats  # noqa: E402

CONFIGS = {
    "prieure_3 (Score 97)": dict(gc=dict(poles=2, period=200, mult=1.41), stochrsi=dict(rsi_len=7, stoch_len=21, smooth_k=3)),
    "prieure_1 (max return)": dict(gc=dict(poles=2, period=144, mult=1.0), stochrsi=dict(rsi_len=7, stoch_len=21, smooth_k=3)),
    "brayvb_v2 default": dict(gc={}, stochrsi={}),
}
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "NEARUSDT", "FETUSDT"]


def spec_for(params, tf):
    s = get_preset("brayvb_v2")
    s["tf"] = tf
    s["indicators"] = {"gc": params["gc"], "stochrsi": params["stochrsi"]}
    return s


def main():
    lines = []
    for tf in ("1d", "4h"):
        for name, p in CONFIGS.items():
            rows = []
            for sym in COINS:
                ctx, _ = make_ctx(sym, tf)
                res, m = run_spec(spec_for(p, tf), ctx.df)
                dr = daily_returns(res.equity, ctx.df.index).reindex(ctx.days).fillna(0.0).to_numpy()
                full = stats(dr)                       # whole in-sample period (what an optimiser shows)
                oos = stats(dr[ctx.oos_mask])          # walk-forward test windows only
                tot = float(np.prod(1 + dr) - 1)
                rows.append((sym, tot, full["sharpe"], full["mdd"], oos["sharpe"], oos["mdd"], int(m.get("n_trades", 0))))
            lines.append(f"\n## {tf} · {name}\n")
            lines.append("| coin | in-sample total | in-sample Sharpe | in-sample maxDD | WF-OOS Sharpe | WF-OOS maxDD | trades |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in rows:
                lines.append(f"| {r[0]} | {r[1]*100:+.0f}% | {r[2]:.2f} | {r[3]*100:.0f}% | {r[4]:.2f} | {r[5]*100:.0f}% | {r[6]} |")
            oo = [r[4] for r in rows]
            lines.append(f"\nmedian WF-OOS Sharpe {np.nanmedian(oo):.2f}; OOS Sharpe > 0 on {sum(x > 0 for x in oo)}/8 coins")
    out = ROOT / "research" / "reports" / "prieure_check.md"
    out.write_text("# 'Prieuré Strategy' configs, independent check\n" + "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
