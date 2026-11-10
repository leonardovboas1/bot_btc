"""Staged search on tiny synthetic data: noise primitives fail DSR/PBO, a planted edge survives S1."""
import numpy as np
import pandas as pd
import pytest

from btclab.indicators.registry import REGISTRY, IndicatorSpec
from btclab.research.search import stages
from btclab.research.search.core import build_context
from btclab.research.search.trials import TrialStore, spec_hash

N_NOISE = 40
N_BARS = 2400


def _register(name, up):
    REGISTRY[name] = IndicatorSpec(
        name=name, func=lambda close, **p: {"up": up(close), "down": ~up(close)}, inputs=("close",),
        defaults={}, space={}, primitives=(f"{name}.up", f"{name}.down"))


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2019-09-01", periods=N_BARS, freq="1D", tz="UTC")
    r = rng.normal(0, 0.01, N_BARS)
    close = 100 * np.cumprod(1 + r)
    opn = np.concatenate([[100.0], close[:-1]])
    df = pd.DataFrame({"open": opn, "close": close, "high": np.maximum(opn, close) * 1.001,
                       "low": np.minimum(opn, close) * 0.999, "volume": 1.0}, index=idx)
    flip = rng.random(N_BARS) < 0.25

    def planted(c):                       # sees next bar's direction, wrong 25% of the time
        nxt = np.concatenate([c[1:] > c[:-1], [False]])
        return nxt ^ flip[: len(c)]

    names = ["planted"] + [f"noise{k}" for k in range(N_NOISE)]
    _register("planted", planted)
    for k in range(N_NOISE):
        _register(f"noise{k}", lambda c, k=k: np.random.default_rng(100 + k).random(len(c)) < 0.5)
    ctx = build_context("1d", df=df, cutoff=idx[-1] + pd.Timedelta(days=1),
                        constraints=dict(min_is_trades=20, min_oos_trades=5))
    yield ctx, names
    for n in names:
        REGISTRY.pop(n, None)


def _prims(names):
    return [p for n in names for p in (f"{n}.up", f"{n}.down")]


def test_hash_ignores_name_and_dedups(synth):
    a = stages.enumerate_s1("1d", ["planted.up"])[0]
    b = dict(a, name="other")
    assert spec_hash(a) == spec_hash(b)
    store = TrialStore(":memory:")
    rec = dict(stage="s1", tf="1d", spec_json="{}", spec_hash=spec_hash(a))
    store.insert_many([rec, rec])
    assert store.count() == 1


def test_planted_edge_survives_s1_and_noise_fails_dsr_pbo(synth):
    ctx, names = synth
    store = TrialStore(":memory:")
    info = stages.run_s1(store, ctx, "1d", n_jobs=1, prims=_prims(names), report=False)
    rows = store.fetch("s1", "1d")
    assert len(rows) == 12 * len(names) and info["n_errors"] == 0
    surv = rows[rows.spec_hash.isin(info["survivors"])]
    assert surv.tag.str.startswith("planted").any(), "planted edge must survive S1"

    mat, hs = store.returns_matrix(tf="1d")
    tags = dict(zip(rows.spec_hash, rows.tag))
    is_noise = np.array([not tags[h].startswith("planted") for h in hs])
    n_total = store.count()

    # best noise candidate: fails DSR (N = all trials) and the noise-only family has PBO ~ 0.5
    noise = mat[:, is_noise]
    best_n = int(np.nanargmax([np.mean(c) / (np.std(c) + 1e-12) for c in noise.T]))
    fs = stages.family_stats(noise, best_n, n_total)
    assert not fs["dsr"].passed
    assert fs["pbo"].pbo > 0.3 and not fs["pbo"].passed

    # planted candidate in the full family: passes DSR
    sh = [np.mean(c) / (np.std(c) + 1e-12) for c in mat.T]
    best = int(np.argmax(sh))
    assert tags[hs[best]].startswith("planted")
    fp = stages.family_stats(mat, best, n_total)
    assert fp["dsr"].passed
