import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btclab.liquidity import depth, liq_model, primitives, service, volume_profile  # noqa: E402
from btclab.liquidity._util import bin_index  # noqa: E402


def make(n=800, seed=3, freq="1h"):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC", name="open_time")
    df = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": rng.lognormal(8, .4, n)}, index=idx)
    mi = pd.date_range("2022-12-31", periods=n * 12 + 300, freq="5min", tz="UTC", name="create_time")
    oi = 50000 * np.exp(np.cumsum(rng.normal(0, 0.002, len(mi))))
    m = pd.DataFrame({"sum_open_interest": oi,
                      "sum_toptrader_long_short_ratio": 1 + rng.normal(0, .2, len(mi))}, index=mi)
    # depth snapshots every 30 s for the first ~20 h only (then a gap)
    ti = pd.date_range("2023-01-01", periods=2400, freq="30s", tz="UTC", name="timestamp")
    rows = []
    for b in range(1, 6):
        rows.append(pd.DataFrame({"percentage": -float(b), "depth": 1.0,
                                  "notional": b * 1e8 * (1 + .1 * np.sin(np.arange(len(ti)) / 50))}, index=ti))
        rows.append(pd.DataFrame({"percentage": float(b), "depth": 1.0, "notional": b * 1e8 * 0.8}, index=ti))
    return df, m, pd.concat(rows).sort_index(kind="stable")


def test_depth_imbalance_asof_and_gap():
    df, m, raw = make()
    snap = depth.depth_snapshots(raw)
    assert snap.shape[1] == 10
    s = depth.depth_imbalance(snap, df.index, "1h", 1)
    assert s.iloc[:20].notna().all()
    assert s.iloc[40:].isna().all()          # > 12 h stale after snapshots end
    assert s.iloc[:20].between(-1, 1).all()
    hm = depth.depth_heatmap(snap, df["close"], "1h")
    assert hm["bid"].shape == hm["ask"].shape and hm["bid"].sum() > 0
    json.dumps(depth.heatmap_json(hm))


def test_liq_levels_clear_and_flush():
    df, m, _ = make()
    oi = liq_model.oi_on_bars(m, df.index, "1h")
    res = liq_model.liq_levels(df["close"], df["high"], df["low"], oi)
    assert res["long"].shape == res["short"].shape and res["long"].sum() > 0
    lo_i = res["bin_lo"]
    for t in (100, 400, 799):
        il = int(bin_index(df["low"].iloc[t]) - lo_i)
        ih = int(bin_index(df["high"].iloc[t]) - lo_i)
        # nothing survives inside the bar's traded range (new levels sit >= ~1/100 away)
        assert res["long"][t][max(il, 0):ih + 1].sum() == 0
        assert res["short"][t][max(il, 0):ih + 1].sum() == 0
    f = liq_model.liq_flush(df["close"], df["high"], df["low"], oi, w=100, tau=1.0)
    assert f.dtype == bool and f.any()


def test_volume_profile():
    df, _, _ = make()
    p = volume_profile.rolling_profile(df["high"], df["low"], df["volume"], 100, nodes=True)
    ok = ~np.isnan(p["poc"])
    assert ok.sum() == len(df) - 99
    assert (p["val"][ok] <= p["poc"][ok] + 1e-6).all() and (p["poc"][ok] <= p["vah"][ok] + 1e-6).all()
    snap = volume_profile.profile_snapshot(df["high"][:100], df["low"][:100], df["volume"][:100])
    json.dumps(snap)
    assert abs(snap["poc"] - p["poc"][99]) < 1e-6


def test_primitives_causality():
    df, m, raw = make()
    snap = depth.depth_snapshots(raw)
    cut = 500
    for name in primitives.REGISTRY:
        kw = dict(w=40)
        if name == "vol_profile":
            kw = dict(window=100)
        full = primitives.compute(name, df, m, snap, **kw)
        part = primitives.compute(name, df.iloc[:cut], m, snap, **kw)
        for k in full:
            assert full[k].dtype == bool
            np.testing.assert_array_equal(full[k][:cut], part[k], err_msg=f"{name}.{k}")


def test_liq_levels_causality():
    df, m, _ = make(400)
    oi = liq_model.oi_on_bars(m, df.index, "1h")
    a = liq_model.liq_levels(df["close"], df["high"], df["low"], oi)
    b = liq_model.liq_levels(df["close"][:300], df["high"][:300], df["low"][:300], oi[:300])
    off = b["bin_lo"] - a["bin_lo"]
    n = b["long"].shape[1]
    assert off >= 0
    for key in ("long", "short"):
        np.testing.assert_allclose(a[key][250][off:off + n], b[key][250], rtol=1e-4, atol=1.0)


def test_service_real_data_smoke():
    out = service.volume_profile_payload("2024-01-01", "2024-01-10", "1h")
    json.dumps(out)
    assert "poc" in out
