"""Amendment 7 design-phase run: 12 configs + benchmarks on the 8 design coins, data < holdout cutoff only.
Writes research/reports/a7_design.md. Informs nothing outside the registered grids."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.a7.runner import table, run_set  # noqa: E402
from research.a7.strategies import DESIGN, DESIGN_END, bench_brayvb, load_panel  # noqa: E402


def main():
    # default holdout enforcement: end = DESIGN_END (< cutoff), allow_holdout stays False
    panel = load_panel(DESIGN, "perp", end=DESIGN_END, need_spot=True)
    br = bench_brayvb(DESIGN, "perp", end=DESIGN_END)
    res = run_set(panel, br, None)
    first = panel.close.index[0] + __import__("pandas").Timedelta(days=120)
    best = max((n for n in res if n[0] == "C"), key=lambda n: res[n]["sharpe"])
    L = ["# A7 design-phase run (design set only)", "",
         f"Set: {', '.join(DESIGN)}; daily bars up to {DESIGN_END.date()} (exclusive; holdout cutoff 2026-03-10 14:00 UTC, "
         f"loaders' default enforcement). Evaluation window {first.date()} -> {(DESIGN_END - __import__('pandas').Timedelta(days=1)).date()} "
         "(first 120 days of the panel excluded for strategies and benchmarks alike).", "",
         "Costs: perp 5 bp fee + 2 bp slippage per side, spot 10 bp + 2 bp, funding at settlement (daily sum). Weights decided at close t earn day t+1. "
         "Spot hedge leg for C3 uses spot klines" + (f" (fallback to perp price for {panel.no_spot})" if panel.no_spot else "") + ".", "",
         "Design-phase look only: it informs nothing outside the registered A7 grids (no config was added or changed).", "",
         table(res), "",
         f"Highest design Sharpe among the 12 configs: **{best}** ({res[best]['sharpe']:.2f}) vs EW B&H {res['EW_BH']['sharpe']:.2f}, "
         f"brayvb {res['BRAYVB']['sharpe']:.2f}. Success rules are applied only on the validation sets."]
    import json
    (ROOT / "research" / "runs" / "a7_design_metrics.json").write_text(json.dumps(
        {n: {k: v for k, v in m.items() if k not in ("r", "turn")} for n, m in res.items()}, default=float))
    out = ROOT / "research" / "reports" / "a7_design.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
