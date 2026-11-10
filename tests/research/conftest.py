"""Shared fixtures for the research/stats tests.

Synthetic families: T = 3000 daily bars, N = 500 trials of iid normal returns
(sigma 2 %/day, zero mean). The planted family adds a constant drift to column
0 so that its true annualised Sharpe is 2 (365 periods/year).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PPY = 365
T = 3000
N = 500
SIGMA = 0.02
SEED = 4
PLANTED_SR_ANNUAL = 2.0


def make_family(seed: int = SEED, planted_sr_annual: float | None = None, t: int = T, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((t, n)) * SIGMA
    if planted_sr_annual is not None:
        m[:, 0] += planted_sr_annual / np.sqrt(PPY) * SIGMA
    return m


@pytest.fixture(scope="session")
def noise_family() -> np.ndarray:
    return make_family()


@pytest.fixture(scope="session")
def planted_family() -> np.ndarray:
    return make_family(planted_sr_annual=PLANTED_SR_ANNUAL)
