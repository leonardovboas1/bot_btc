import numpy as np
import pytest
from scipy.stats import norm

from btclab.research.stats.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    psr_from_returns,
    return_moments,
    sharpe_ratio,
    trial_sharpes,
)


def test_psr_hand_computation_normal():
    # SR = 0.1 per period, T = 101, normal returns (skew 0, kurt 3), SR* = 0:
    # z = 0.1 * sqrt(100) / sqrt(1 + (3-1)/4 * 0.01) = 1 / sqrt(1.005) = 0.99750934
    # PSR = Phi(0.99750934) = 0.84074133
    assert probabilistic_sharpe_ratio(0.1, 101, 0.0, 0.0, 3.0) == pytest.approx(0.8407413278, abs=1e-9)


def test_psr_hand_computation_non_normal():
    # SR = 0.05, T = 1001, skew = -1, kurt = 5, SR* = 0.01:
    # radicand = 1 - (-1)(0.05) + (5-1)/4 * 0.0025 = 1.0525
    # z = 0.04 * sqrt(1000) / sqrt(1.0525) = 1.23295986
    # PSR = Phi(1.23295986) = 0.89120463
    assert probabilistic_sharpe_ratio(0.05, 1001, 0.01, -1.0, 5.0) == pytest.approx(0.8912046304, abs=1e-9)
    # negative skew / fat tails lower the PSR relative to the normal case
    assert probabilistic_sharpe_ratio(0.05, 1001, 0.01, -1.0, 5.0) < probabilistic_sharpe_ratio(0.05, 1001, 0.01, 0.0, 3.0)


def test_psr_from_returns_matches_formula():
    rng = np.random.default_rng(0)
    r = rng.standard_t(5, 2000) * 0.01 + 0.0004
    n, sr, skew, kurt = return_moments(r)
    expected = norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))
    assert psr_from_returns(r) == pytest.approx(expected, abs=1e-12)
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert sharpe_ratio(r, 365) == pytest.approx(sr * np.sqrt(365))


def test_min_trl_inverts_psr():
    # at T = MinTRL the PSR equals alpha
    sr, skew, kurt = 0.05, -1.0, 5.0
    trl = min_track_record_length(sr, 0.0, skew, kurt, alpha=0.95)
    assert trl == pytest.approx(1140.0338, abs=1e-3)
    assert probabilistic_sharpe_ratio(sr, trl, 0.0, skew, kurt) == pytest.approx(0.95, abs=1e-9)
    assert min_track_record_length(0.0, 0.0) == float("inf")


def test_expected_max_sharpe():
    # E[max] grows with N and with the trial-Sharpe dispersion; zero for a single trial
    assert expected_max_sharpe(1, 0.01) == 0.0
    e10 = expected_max_sharpe(10, 0.01)
    e1000 = expected_max_sharpe(1000, 0.01)
    assert 0 < e10 < e1000
    assert expected_max_sharpe(1000, 0.04) == pytest.approx(2 * e1000)
    # hand value: N = 100, V = 1 -> 0.4228 * ppf(0.99) + 0.5772 * ppf(1 - 1/(100 e))
    hand = (1 - 0.5772156649) * norm.ppf(0.99) + 0.5772156649 * norm.ppf(1 - 1 / (100 * np.e))
    assert expected_max_sharpe(100, 1.0) == pytest.approx(hand, abs=1e-9)


def test_dsr_deflates_with_more_trials():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(1500) * 0.02 + 0.0015
    res1 = deflated_sharpe_ratio(r, n_trials=1, var_trial_sharpes=1 / 1500)
    res500 = deflated_sharpe_ratio(r, n_trials=500, var_trial_sharpes=1 / 1500, periods_per_year=365)
    assert res1.dsr == pytest.approx(res1.psr)
    assert res500.dsr < res1.dsr
    assert res500.sr_annualised == pytest.approx(res500.sr * np.sqrt(365))
    d = res500.to_dict()
    assert {"dsr", "psr", "expected_max_sr", "min_trl", "passed"} <= set(d)


def test_trial_sharpes_matrix():
    rng = np.random.default_rng(2)
    m = rng.standard_normal((400, 7)) * 0.01
    ts = trial_sharpes(m)
    assert ts.shape == (7,)
    assert ts[3] == pytest.approx(m[:, 3].mean() / m[:, 3].std(ddof=1))
