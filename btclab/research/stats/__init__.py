"""Statistical anti-overfitting toolkit (plan.md section C).

dsr        Probabilistic / Deflated Sharpe Ratio, MinTRL
cscv       Probability of Backtest Overfitting (CSCV)
spa        White's Reality Check + Hansen's SPA (stationary bootstrap)
bootstrap  Block bootstrap of trade / daily returns
walkforward Purged + embargoed walk-forward folds, holdout cutoff
stability  Parameter-neighbourhood stability
regimes    Causal regime labels and per-regime performance
gauntlet   run_gauntlet: all pass criteria in one report
"""
from .bootstrap import BootstrapResult, block_bootstrap, circular_block_indices, stationary_bootstrap_indices
from .cscv import CSCVResult, cscv_pbo
from .dsr import (
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    psr_from_returns,
    sharpe_ratio,
    trial_sharpes,
)
from .gauntlet import DEFAULT_CRITERIA, GauntletReport, TestResult, run_gauntlet
from .regimes import REGIME_LABELS, label_regimes, regime_performance
from .spa import SPAResult, long_run_variance, spa_test
from .stability import StabilityResult, parameter_stability, stability_at_peak
from .walkforward import Fold, holdout_cutoff, validate_folds, walk_forward_folds

__all__ = [
    "BootstrapResult", "block_bootstrap", "circular_block_indices", "stationary_bootstrap_indices",
    "CSCVResult", "cscv_pbo",
    "DSRResult", "deflated_sharpe_ratio", "expected_max_sharpe", "min_track_record_length",
    "probabilistic_sharpe_ratio", "psr_from_returns", "sharpe_ratio", "trial_sharpes",
    "DEFAULT_CRITERIA", "GauntletReport", "TestResult", "run_gauntlet",
    "REGIME_LABELS", "label_regimes", "regime_performance",
    "SPAResult", "long_run_variance", "spa_test",
    "StabilityResult", "parameter_stability", "stability_at_peak",
    "Fold", "holdout_cutoff", "validate_folds", "walk_forward_folds",
]
