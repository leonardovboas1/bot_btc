import numpy as np
import pandas as pd

from btclab.signals.combiner import heikin_ashi


def _ref(df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hc = (o + h + l + c) / 4
    ho = np.empty(len(o)); ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(o)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return ho, hc, np.maximum.reduce([h, ho, hc]), np.minimum.reduce([l, ho, hc])


def test_matches_loop_and_is_causal():
    rng = np.random.default_rng(0)
    c = 100 + rng.normal(0, 1, 500).cumsum()
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5,
                       "close": c, "volume": 1.0})
    ha = heikin_ashi(df)
    for got, want in zip((ha.open, ha.close, ha.high, ha.low), _ref(df)):
        assert np.allclose(got.to_numpy(), want)
    part = heikin_ashi(df.iloc[:300])
    assert np.allclose(part.to_numpy(), ha.iloc[:300].to_numpy())
