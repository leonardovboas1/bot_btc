import numpy as np
import pandas as pd
import pytest

from btclab.research.stats.stability import parameter_stability, stability_at_peak


def _grid(score_fn):
    rows = [(fast, slow, score_fn(fast, slow)) for fast in range(5, 41, 5) for slow in range(20, 201, 20)]
    return pd.DataFrame(rows, columns=["fast", "slow", "score"])


def test_neighbours_within_30pct_each_param():
    df = _grid(lambda f, s: 1.0)
    out = parameter_stability(df, ["fast", "slow"], "score", tol=0.3)
    row = out[(out.fast == 20) & (out.slow == 100)].iloc[0]
    # fast in [14, 26] -> {15, 20, 25}; slow in [70, 130] -> {80, 100, 120}; minus self = 8
    assert row.n_neighbours == 8
    assert row.stability_ratio == pytest.approx(1.0)
    assert "neighbour_min" in out.columns


def test_smooth_surface_is_stable_and_spike_is_not():
    smooth = _grid(lambda f, s: 2.0 - 0.01 * abs(f - 20) - 0.002 * abs(s - 100))
    res = stability_at_peak(smooth, ["fast", "slow"], "score", threshold=0.7)
    assert res.peak_params == {"fast": 20, "slow": 100}
    assert res.stability_ratio > 0.9 and res.passed

    spiky = _grid(lambda f, s: 0.3)
    spiky.loc[(spiky.fast == 20) & (spiky.slow == 100), "score"] = 3.0
    res2 = stability_at_peak(spiky, ["fast", "slow"], "score", threshold=0.7)
    assert res2.stability_ratio == pytest.approx(0.1)
    assert not res2.passed


def test_no_neighbours_gives_nan_and_fails():
    df = pd.DataFrame({"p": [1.0, 10.0, 100.0], "score": [1.0, 2.0, 3.0]})
    res = stability_at_peak(df, ["p"], "score")
    assert res.n_neighbours == 0 and np.isnan(res.stability_ratio) and not res.passed
