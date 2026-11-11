"""Builds a8_design.md + a8_candidates.json from the design jsonl; runs T3 (long-only / short-only) variants of the advancing 10."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.a8.core import ROOT, DESIGN, aggregate, eval_coin, gates  # noqa: E402
from research.a8 import templates as T  # noqa: E402
from research.a8.design import fam, record  # noqa: E402

PT = {
    "macross.close_above_slow": "preco fecha acima da media movel lenta", "macross.close_below_slow": "preco fecha abaixo da media movel lenta",
    "macross.fast_above_slow": "media rapida acima da lenta", "macross.fast_below_slow": "media rapida abaixo da lenta",
    "supertrend.up": "Supertrend em alta", "supertrend.down": "Supertrend em baixa",
    "ichimoku.above_cloud": "preco acima da nuvem Ichimoku", "ichimoku.below_cloud": "preco abaixo da nuvem Ichimoku",
    "dmi.bull_strong": "ADX forte com DI+ > DI-", "dmi.bear_strong": "ADX forte com DI- > DI+",
    "gc.filter_rising": "Gaussian Channel com filtro subindo", "gc.filter_falling": "Gaussian Channel com filtro caindo",
    "macd.cross_up": "MACD cruza acima do sinal", "macd.cross_down": "MACD cruza abaixo do sinal",
    "macd.hist_rising": "histograma MACD subindo", "not:macd.hist_rising": "histograma MACD nao subindo",
    "mfi.cross_up_os": "MFI (proxy do RSI) sai da sobrevenda", "mfi.cross_down_ob": "MFI (proxy do RSI) sai da sobrecompra",
    "stochrsi.k_cross_up_d": "Stoch RSI: K cruza acima de D", "stochrsi.k_cross_down_d": "Stoch RSI: K cruza abaixo de D",
    "stochrsi.bull_cross_os": "Stoch RSI: K cruza D para cima na sobrevenda", "stochrsi.bear_cross_ob": "Stoch RSI: K cruza D para baixo na sobrecompra",
    "cipherb.wt_cross_up": "Cipher B: WaveTrend cruza para cima", "cipherb.wt_cross_down": "Cipher B: WaveTrend cruza para baixo",
    "qqe.cross_up": "QQE cruza para cima", "qqe.cross_down": "QQE cruza para baixo",
    "bb.cross_above_basis": "preco cruza acima da media das Bollinger", "bb.cross_below_basis": "preco cruza abaixo da media das Bollinger",
    "bb.cross_above_upper": "preco cruza acima da banda superior", "bb.cross_below_lower": "preco cruza abaixo da banda inferior",
}


def pt(p):
    if isinstance(p, list):
        return "(" + " OU ".join(pt(x) for x in p) + ")"
    return PT.get(p, p.replace("_", " "))


def describe(spec, tag=""):
    def side(k):
        v = spec.get(k)
        return " E ".join(pt(x) for x in v) if v else None
    L = []
    for k, lab in (("long", "COMPRA (long) quando"), ("short", "VENDE a descoberto (short) quando"),
                   ("exit_long", "sai do long quando"), ("exit_short", "sai do short quando")):
        s = side(k)
        if s:
            L.append(f"{lab}: {s}")
    r = spec.get("risk") or {}
    if r:
        L.append("stop: " + ", ".join(f"{k}={v}" for k, v in r.items()) + " (multiplos de ATR)")
    return "; ".join(L)


def main():
    rows = [json.loads(l) for l in (ROOT / "research/runs/a8_design_combos.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "best" in r]
    ok.sort(key=lambda r: -r["best"]["score"])
    passing = [r for r in ok if r["gates"]["all"]]
    adv = passing[:10]
    cand = []
    L = ["# A8 design: indicator long/short strategies, one shared parameter set across the 8 design coins", ""]
    n_trials = sum(r["n_trials"] for r in rows)
    pr = sum(r["pruned_early"] for r in rows)
    L += [f"Combos run: {len(rows)} (4h and 1d; 1h skipped, see below); Optuna trials: {n_trials} (all recorded in `research/trials.duckdb`, stage `a8`); "
          f"combos pruned after 20 trials (best training Sharpe < 0): {pr}; combos passing all gates: {len(passing)}.", "",
          "Selection per combo: trial with best training objective (median over coins of mean in-sample fold Sharpe). Score = median over coins of stitched "
          "OOS Sharpe of that trial. Gates: OOS Sharpe > 0 on >= 6/8 coins; pooled OOS profit factor > 1.1; short-leg pooled PnL >= 0 or its loss <= 20% "
          "of the long leg's gross loss; >= 30 OOS trades on every coin (strict, also on 1d). Costs BINANCE_PERP_TAKER + funding; 6 WF folds per coin "
          "(coins with < 12 months training dropped from that fold, as in A5). 1h was not run: the 25 fresh V1 coins have no 1h data on disk, so 1h "
          "candidates could not be validated; the T1/T2 grids were run on 4h and 1d only.", "",
          "## Indicator mapping (named indicator -> registry primitive)", "", "```", T.__doc__.split("MAPPING")[1].split('"""')[0].strip("() \n"), "```", "",
          "## Top 30 by score (all combos; gates column: pos6/8, PF, side, trades)", "",
          "| # | combo | score (med OOS Sharpe) | train obj | coins>0 | PF | short pnl | long pnl | min trades | gates |", "|" + "---|" * 10]
    for i, r in enumerate(ok[:30], 1):
        b, g = r["best"], r["gates"]
        gs = "ALL" if g["all"] else ",".join(k for k in ("pos", "pf", "side", "trades") if not g[k]) + " fail"
        L.append(f"| {i} | {fam(r['combo'])[3:]} | {b['score']:.2f} | {b['obj']:.2f} | {b['n_pos']}/8 | {b['pf']:.2f} | "
                 f"{b['short_pnl']:.2f} | {b['long_pnl']:.2f} | {b['min_trades']} | {gs} |")
    L += ["", f"## Advancing candidates ({len(adv)}: top by score among combos passing all gates)", ""]
    t3 = []
    for i, r in enumerate(adv, 1):
        c, spec = r["combo"], r["spec"]
        name = f"a8c{i}"
        spec = dict(spec, name=name)
        b = r["best"]
        res = {}
        for sides in ("long", "short"):
            s3 = T.build(c, r["params"], sides, name=f"{name}_{sides}")
            a = aggregate([eval_coin(s3, s, c["tf"]) for s in DESIGN])
            res[sides] = a
            t3.append(record(c, s3, a, tag="T3"))
        cand.append(dict(rank=i, name=name, combo=c, params=r["params"], spec=spec, design=b,
                         t3={k: dict(score=v["score"], pf=v["pf"], pnl=v["long_pnl"] + v["short_pnl"]) for k, v in res.items()}))
        L += [f"### {i}. {name}: {fam(c)[3:]}", "",
              f"- Regra: {describe(spec)}.", f"- Parametros: `{json.dumps(r['params'], default=float)}`",
              f"- Design: score {b['score']:.2f}, treino {b['obj']:.2f}, {b['n_pos']}/8 moedas > 0, PF {b['pf']:.2f}, short pnl {b['short_pnl']:.2f} / long pnl {b['long_pnl']:.2f}, "
              f"Sharpe EW OOS {b['pool_sharpe']:.2f}, maxDD EW {b['pool_dd']*100:.0f}%. Por moeda: {b['per_coin']}",
              f"- T3 (so long / so short, OOS med. Sharpe): long {res['long']['score']:.2f} (PF {res['long']['pf']:.2f}), short {res['short']['score']:.2f} (PF {res['short']['pf']:.2f})", ""]
    if not adv:
        L.append("No combo passed all gates.")
    if t3:
        from btclab.research.search.trials import TrialStore
        st = TrialStore()
        st.insert_many(t3)
        st.close()
    (ROOT / "research/runs/a8_candidates.json").write_text(json.dumps(cand, indent=1, default=float))
    (ROOT / "research/reports/a8_design.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]), len(adv))


if __name__ == "__main__":
    main()
