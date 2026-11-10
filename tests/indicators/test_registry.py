import numpy as np
import pytest

from btclab.indicators.registry import REGISTRY, all_primitives, compute, warmup


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_primitives_are_bool_outputs(name, ohlcv):
    spec = REGISTRY[name]
    out = compute(name, ohlcv)
    for prim in spec.primitives:
        ns, key = prim.split(".", 1)
        assert ns == name
        assert key in out, prim
        assert out[key].dtype == bool, prim
        assert len(out[key]) == len(ohlcv)
    for k, v in out.items():
        assert v.dtype in (np.float64, np.bool_), (name, k, v.dtype)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_defaults_inside_space(name):
    spec = REGISTRY[name]
    assert spec.constraint(spec.defaults)
    for p, sp in spec.space.items():
        assert p in spec.defaults, (name, p)
        v = spec.defaults[p]
        if sp["type"] == "cat":
            assert v in sp["choices"]
        else:
            assert sp["low"] <= v <= sp["high"], (name, p)
    assert warmup(name) >= 0


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_space_corners_run(name, synth):
    spec = REGISTRY[name]
    for corner in ("low", "high"):
        p = {k: (sp["choices"][-1] if sp["type"] == "cat" else sp[corner]) for k, sp in spec.space.items()}
        if not spec.constraint({**spec.defaults, **p}):
            continue
        out = compute(name, synth.iloc[:800], **p)
        assert all(len(v) == 800 for v in out.values())


def test_primitive_names_unique():
    prims = all_primitives()
    assert len(prims) == len(set(prims))
    for must in ("gc.filter_rising", "gc.close_above_hband", "cipherb.buy", "cipherb.bull_div"):
        assert must in prims
