import time
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from btclab.research.stats.cscv import cscv_pbo, sharpe_metric


def _reference_pbo(m: np.ndarray, n_blocks: int):
    """Plain-loop CSCV (Sharpe) used to validate the vectorised fast path."""
    t = m.shape[0]
    bounds = np.linspace(0, t, n_blocks + 1).astype(int)
    row_block = np.repeat(np.arange(n_blocks), np.diff(bounds))
    logits, is_b, oos_b = [], [], []
    for combo in combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.isin(row_block, combo)
        is_m = sharpe_metric(m[is_rows])
        oos_m = sharpe_metric(m[~is_rows])
        n_star = int(is_m.argmax())
        rank = 1 + (oos_m < oos_m[n_star]).sum()
        w = rank / (m.shape[1] + 1)
        logits.append(np.log(w / (1 - w)))
        is_b.append(is_m[n_star]); oos_b.append(oos_m[n_star])
    logits = np.array(logits)
    return float((logits <= 0).mean()), logits, np.array(is_b), np.array(oos_b)


def test_fast_path_matches_reference_loop():
    rng = np.random.default_rng(3)
    m = rng.standard_normal((400, 25)) * 0.01
    m[:, 4] += 0.002
    res = cscv_pbo(m, n_blocks=8, metric="sharpe")
    pbo_ref, logits_ref, is_ref, oos_ref = _reference_pbo(m, 8)
    assert res.n_splits == 70
    assert res.pbo == pytest.approx(pbo_ref)
    np.testing.assert_allclose(res.logits, logits_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(res.is_metric_best, is_ref, rtol=1e-9)
    np.testing.assert_allclose(res.oos_metric_best, oos_ref, rtol=1e-9)


def test_callable_metric_matches_string_metric():
    rng = np.random.default_rng(5)
    m = rng.standard_normal((320, 12)) * 0.01
    a = cscv_pbo(m, n_blocks=6, metric="sharpe")
    b = cscv_pbo(m, n_blocks=6, metric=sharpe_metric)
    np.testing.assert_allclose(a.logits, b.logits, rtol=1e-9, atol=1e-9)
    c = cscv_pbo(pd.DataFrame(m), n_blocks=6, metric="mean")
    d = cscv_pbo(m, n_blocks=6, metric=lambda sub: sub.mean(axis=0))
    np.testing.assert_allclose(c.logits, d.logits, rtol=1e-9, atol=1e-9)


def test_degradation_and_prob_loss_fields():
    rng = np.random.default_rng(6)
    m = rng.standard_normal((640, 30)) * 0.01
    res = cscv_pbo(m, n_blocks=8)
    assert np.isfinite(res.degradation_slope)
    assert 0.0 <= res.prob_loss <= 1.0
    assert set(res.to_dict()) >= {"pbo", "degradation_slope", "degradation_r2", "prob_loss", "passed"}


def test_runtime_c16_8_on_1000_trials():
    rng = np.random.default_rng(8)
    m = rng.standard_normal((3000, 1000)) * 0.02
    t0 = time.perf_counter()
    res = cscv_pbo(m, n_blocks=16)
    elapsed = time.perf_counter() - t0
    assert res.n_splits == 12870
    assert elapsed < 20.0, f"CSCV took {elapsed:.1f}s"


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((100, 5)), n_blocks=5)
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros(100))
