"""The statistical gauntlet: apply the pass criteria of plan.md section C.

| Test                       | Pass                                              |
|----------------------------|---------------------------------------------------|
| WF OOS Sharpe              | > 0 in >= 4 of 6 folds                            |
| Deflated Sharpe (N trials) | DSR > 0.95                                        |
| PBO (CSCV, S = 16)         | < 0.2                                             |
| White RC + Hansen SPA      | p < 0.05 vs buy-and-hold AND vs zero              |
| Block bootstrap (10k)      | 5th pct total return > 0 and > B&H 5th pct        |
| Param stability ±30 %      | neighbour median >= 70 % of peak                  |
| Regimes                    | no regime worse than B&H (mean daily return)      |
| Holdout                    | report only                                       |
| Objective constraints      | >= 100 IS trades, >= 30 per OOS fold, MaxDD <= 35 %, PF > 1.2 |

Every input beyond ``returns`` is optional: a test whose inputs are missing is
reported as ``skipped`` and does not fail the gauntlet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .bootstrap import block_bootstrap
from .cscv import cscv_pbo
from .dsr import deflated_sharpe_ratio, sharpe_ratio, trial_sharpes
from .regimes import label_regimes, regime_performance
from .spa import spa_test
from .stability import stability_at_peak

__all__ = ["run_gauntlet", "GauntletReport", "TestResult", "DEFAULT_CRITERIA"]

DEFAULT_CRITERIA: Dict[str, Any] = {
    "wf_min_positive_folds": 4,
    "wf_n_folds": 6,
    "dsr_threshold": 0.95,
    "pbo_threshold": 0.20,
    "pbo_blocks": 16,
    "spa_threshold": 0.05,
    "spa_n_boot": 2000,
    "spa_mean_block": 10.0,
    "bootstrap_paths": 10_000,
    "bootstrap_percentile": 5,
    "stability_tol": 0.30,
    "stability_threshold": 0.70,
    "regime_compare_on": "mean",
    "regime_tolerance": 0.0,
    "min_is_trades": 100,
    "min_oos_trades_per_fold": 30,
    "max_drawdown": 0.35,
    "min_profit_factor": 1.2,
}


@dataclass
class TestResult:
    name: str
    value: Any
    threshold: Any
    passed: Optional[bool]          # None = skipped
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "skipped" if self.passed is None else ("pass" if self.passed else "FAIL")

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "threshold": self.threshold,
                "passed": self.passed, "status": self.status, "detail": self.detail}


@dataclass
class GauntletReport:
    results: Dict[str, TestResult]
    passed: bool                    # all executed tests passed
    n_passed: int
    n_failed: int
    n_skipped: int
    holdout: Optional[Dict[str, Any]] = None   # report only

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "holdout": self.holdout,
        }

    def summary(self) -> str:
        lines = [f"GAUNTLET {'PASS' if self.passed else 'FAIL'}  ({self.n_passed} pass / {self.n_failed} fail / {self.n_skipped} skipped)"]
        for r in self.results.values():
            val = r.value if not isinstance(r.value, float) else f"{r.value:.4g}"
            lines.append(f"  [{r.status:>7}] {r.name:<22} value={val}  threshold={r.threshold}")
        if self.holdout:
            lines.append(f"  [ report] holdout                {self.holdout}")
        return "\n".join(lines)


def _series(x, index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    s = pd.Series(np.asarray(x, dtype=float))
    if index is not None and len(index) == len(s):
        s.index = index
    return s


def run_gauntlet(
    returns,
    *,
    trial_returns=None,
    n_trials: Optional[int] = None,
    benchmark_returns=None,
    oos_fold_returns: Optional[Sequence] = None,
    trade_returns=None,
    benchmark_trade_returns=None,
    param_grid: Optional[pd.DataFrame] = None,
    param_cols: Optional[Sequence[str]] = None,
    score_col: Optional[str] = None,
    peak_index: Optional[int] = None,
    close: Optional[pd.Series] = None,
    regimes: Optional[pd.Series] = None,
    holdout_returns=None,
    n_is_trades: Optional[int] = None,
    oos_fold_trade_counts: Optional[Sequence[int]] = None,
    max_drawdown: Optional[float] = None,
    profit_factor: Optional[float] = None,
    periods_per_year: float = 365.0,
    criteria: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> GauntletReport:
    """Run every applicable gauntlet test on one candidate strategy.

    Parameters
    ----------
    returns : daily returns of the candidate (full in-sample/WF history)
    trial_returns : (T x N) daily returns of *all* trials of the family
        (candidate included) -> DSR (via trial Sharpes) and PBO. If only the
        trial count is known pass ``n_trials`` (DSR then uses the candidate's
        own Sharpe variance proxy 1/T, a weaker deflation).
    benchmark_returns : daily buy-and-hold returns aligned with ``returns``
    oos_fold_returns : list of per-fold OOS daily return arrays
    trade_returns : per-trade returns of the candidate (block bootstrap)
    benchmark_trade_returns : per-trade or daily returns of B&H for the bootstrap
        comparison (defaults to ``benchmark_returns``)
    param_grid, param_cols, score_col, peak_index : stability input
    close : daily close (regimes are derived causally) or pass ``regimes``
    holdout_returns : holdout daily returns -> report only
    n_is_trades, oos_fold_trade_counts, max_drawdown, profit_factor : objective constraints
    """
    crit = {**DEFAULT_CRITERIA, **(criteria or {})}
    r = _series(returns)
    res: Dict[str, TestResult] = {}

    # 1. walk-forward OOS Sharpe
    if oos_fold_returns is not None:
        srs = [sharpe_ratio(np.asarray(f, dtype=float), periods_per_year) for f in oos_fold_returns]
        n_pos = int(sum(1 for s in srs if np.isfinite(s) and s > 0))
        res["wf_oos_sharpe"] = TestResult(
            "wf_oos_sharpe", n_pos, f">= {crit['wf_min_positive_folds']} of {len(srs)} folds > 0",
            n_pos >= crit["wf_min_positive_folds"], {"fold_sharpes": srs},
        )
    else:
        res["wf_oos_sharpe"] = TestResult("wf_oos_sharpe", None, None, None)

    # 2. DSR
    if trial_returns is not None:
        ts = trial_sharpes(trial_returns)
        dsr = deflated_sharpe_ratio(r.to_numpy(), ts, periods_per_year=periods_per_year,
                                    threshold=crit["dsr_threshold"])
    elif n_trials is not None:
        dsr = deflated_sharpe_ratio(r.to_numpy(), n_trials=n_trials, var_trial_sharpes=1.0 / max(len(r), 2),
                                    periods_per_year=periods_per_year, threshold=crit["dsr_threshold"])
    else:
        dsr = None
    if dsr is not None:
        res["dsr"] = TestResult("dsr", dsr.dsr, f"> {crit['dsr_threshold']}", dsr.passed, dsr.to_dict())
    else:
        res["dsr"] = TestResult("dsr", None, None, None)

    # 3. PBO
    if trial_returns is not None and np.asarray(trial_returns).shape[1] >= 2:
        pbo = cscv_pbo(trial_returns, n_blocks=crit["pbo_blocks"], threshold=crit["pbo_threshold"])
        res["pbo"] = TestResult("pbo", pbo.pbo, f"< {crit['pbo_threshold']}", pbo.passed, pbo.to_dict())
    else:
        res["pbo"] = TestResult("pbo", None, None, None)

    # 4. RC + SPA vs zero and vs B&H (the candidate alone, or the whole family if given)
    spa_input = trial_returns if trial_returns is not None else r.to_numpy()
    spa_zero = spa_test(spa_input, 0.0, n_boot=crit["spa_n_boot"], mean_block=crit["spa_mean_block"],
                        seed=seed, threshold=crit["spa_threshold"])
    res["spa_vs_zero"] = TestResult("spa_vs_zero", spa_zero.spa_pvalue, f"< {crit['spa_threshold']} (RC and SPA)",
                                    spa_zero.passed, spa_zero.to_dict())
    if benchmark_returns is not None:
        b = _series(benchmark_returns, r.index)
        spa_bh = spa_test(spa_input, b.to_numpy(), n_boot=crit["spa_n_boot"], mean_block=crit["spa_mean_block"],
                          seed=seed, threshold=crit["spa_threshold"])
        res["spa_vs_bh"] = TestResult("spa_vs_bh", spa_bh.spa_pvalue, f"< {crit['spa_threshold']} (RC and SPA)",
                                      spa_bh.passed, spa_bh.to_dict())
    else:
        res["spa_vs_bh"] = TestResult("spa_vs_bh", None, None, None)

    # 5. block bootstrap
    boot_input = trade_returns if trade_returns is not None else r.to_numpy()
    q = crit["bootstrap_percentile"]
    bs = block_bootstrap(boot_input, n_paths=crit["bootstrap_paths"], seed=seed,
                         periods_per_year=None if trade_returns is not None else periods_per_year)
    p_strat = float(np.percentile(bs.total_return, q))
    bench_boot = benchmark_trade_returns if benchmark_trade_returns is not None else benchmark_returns
    detail = {"pct_total_return": p_strat, "pct_max_drawdown": float(np.percentile(bs.max_drawdown, 100 - q)),
              "prob_loss": bs.prob_loss, "percentiles": bs.percentiles.to_dict()}
    if bench_boot is not None:
        bb = block_bootstrap(bench_boot, n_paths=crit["bootstrap_paths"], seed=seed)
        p_bench = float(np.percentile(bb.total_return, q))
        detail["bench_pct_total_return"] = p_bench
        passed = p_strat > 0 and p_strat > p_bench
        thr = f"{q}th pct total return > 0 and > B&H ({p_bench:.4g})"
    else:
        passed = p_strat > 0
        thr = f"{q}th pct total return > 0"
    res["bootstrap"] = TestResult("bootstrap", p_strat, thr, bool(passed), detail)

    # 6. parameter stability
    if param_grid is not None and param_cols and score_col:
        st = stability_at_peak(param_grid, param_cols, score_col, tol=crit["stability_tol"],
                               threshold=crit["stability_threshold"], peak_index=peak_index)
        res["stability"] = TestResult("stability", st.stability_ratio, f">= {crit['stability_threshold']}",
                                      st.passed, st.to_dict())
    else:
        res["stability"] = TestResult("stability", None, None, None)

    # 7. regimes
    if (close is not None or regimes is not None) and benchmark_returns is not None:
        reg = regimes if regimes is not None else label_regimes(close)
        b = _series(benchmark_returns, r.index)
        perf = regime_performance(r, reg, b, periods_per_year, compare_on=crit["regime_compare_on"])
        body = perf.drop(index="all", errors="ignore")
        worst = float(body["excess"].min()) if len(body) else float("nan")
        n_worse = int((body["excess"] < -crit["regime_tolerance"]).sum())
        res["regimes"] = TestResult("regimes", n_worse, "0 regimes worse than B&H", n_worse == 0,
                                    {"worst_excess": worst, "table": perf.to_dict(orient="index")})
    else:
        res["regimes"] = TestResult("regimes", None, None, None)

    # 8. objective constraints (only those supplied)
    if n_is_trades is not None:
        res["min_is_trades"] = TestResult("min_is_trades", n_is_trades, f">= {crit['min_is_trades']}",
                                          n_is_trades >= crit["min_is_trades"])
    if oos_fold_trade_counts is not None:
        mn = int(min(oos_fold_trade_counts)) if len(oos_fold_trade_counts) else 0
        res["min_oos_trades"] = TestResult("min_oos_trades", mn, f">= {crit['min_oos_trades_per_fold']} per fold",
                                           mn >= crit["min_oos_trades_per_fold"], {"counts": list(oos_fold_trade_counts)})
    if max_drawdown is not None:
        res["max_drawdown"] = TestResult("max_drawdown", float(max_drawdown), f"<= {crit['max_drawdown']}",
                                         float(max_drawdown) <= crit["max_drawdown"])
    if profit_factor is not None:
        res["profit_factor"] = TestResult("profit_factor", float(profit_factor), f"> {crit['min_profit_factor']}",
                                          float(profit_factor) > crit["min_profit_factor"])

    # 9. holdout: report only
    holdout = None
    if holdout_returns is not None:
        h = _series(holdout_returns)
        eq = np.cumprod(1.0 + h.to_numpy())
        holdout = {
            "n_obs": int(h.size),
            "sharpe": sharpe_ratio(h, periods_per_year),
            "total_return": float(eq[-1] - 1.0) if h.size else float("nan"),
            "max_drawdown": float((1.0 - eq / np.maximum.accumulate(eq)).max()) if h.size else float("nan"),
        }

    executed = [t for t in res.values() if t.passed is not None]
    n_pass = sum(1 for t in executed if t.passed)
    n_fail = len(executed) - n_pass
    return GauntletReport(
        results=res,
        passed=n_fail == 0 and len(executed) > 0,
        n_passed=n_pass,
        n_failed=n_fail,
        n_skipped=len(res) - len(executed),
        holdout=holdout,
    )
