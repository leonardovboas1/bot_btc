import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btclab.data import loaders  # noqa: E402
from btclab.features import h7  # noqa: E402

HOUR = pd.Timedelta("1h")


def synth(n=1500, seed=1, gap=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    opn = np.r_[close[0], close[:-1]]
    hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = rng.uniform(900, 1100, n)
    perp = pd.DataFrame({"open": opn, "high": hi, "low": lo, "close": close, "volume": vol,
                         "taker_buy_base": vol * rng.uniform(0.4, 0.6, n)}, index=idx)
    spot = pd.Series(close * (1 + rng.normal(0, 0.0005, n)), index=idx)
    fidx = pd.date_range(idx[0], periods=n // 8, freq="8h", tz="UTC")
    fund = pd.Series(rng.normal(1e-4, 5e-5, len(fidx)), index=fidx)
    midx = pd.date_range(idx[0], periods=n * 12, freq="5min", tz="UTC")
    m = pd.DataFrame({
        "sum_open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.001, len(midx)))),
        "count_long_short_ratio": rng.uniform(0.8, 1.6, len(midx)),
        "sum_toptrader_long_short_ratio": rng.uniform(0.8, 1.6, len(midx)),
        "sum_taker_long_short_vol_ratio": np.exp(rng.normal(0, 0.3, len(midx))),
    }, index=midx)
    if gap is not None:
        m = m[(m.index < gap[0]) | (m.index >= gap[1])]
    return perp, spot, fund, m


def build(perp, spot, fund, m):
    return h7.build_features(perp, spot, fund, m, "ETHUSDT", start=perp.index[0])


def test_causality_truncation():
    perp, spot, fund, m = synth()
    full = build(perp, spot, fund, m)
    cut = perp.index[1000]
    t = build(perp[perp.index < cut], spot[spot.index < cut], fund[fund.index < cut], m[m.index < cut])
    a = full.loc[t.index]
    pd.testing.assert_frame_equal(a, t, check_exact=False, rtol=1e-9, atol=1e-12)


def test_causality_future_perturbation():
    perp, spot, fund, m = synth()
    base = build(perp, spot, fund, m)
    cut = perp.index[900]
    p2, s2, f2, m2 = perp.copy(), spot.copy(), fund.copy(), m.copy()
    p2.loc[p2.index >= cut, ["open", "high", "low", "close"]] *= 1.37
    s2[s2.index >= cut] *= 0.9
    f2[f2.index >= cut] = 0.01
    m2.loc[m2.index >= cut, :] *= 2.0
    alt = build(p2, s2, f2, m2)
    cols = [c for c in base.columns]
    pd.testing.assert_frame_equal(base.loc[base.index < cut, cols], alt.loc[alt.index < cut, cols],
                                  check_exact=False, rtol=1e-9, atol=1e-12)


def test_no_holdout_rows():
    cutoff = loaders.get_cutoff()
    assert h7.load_metrics("BTCUSDT").index.max() < cutoff
    for s in h7.SYMBOLS:
        for name in (f"h7_{s}_1h.parquet", f"h7_{s}_1h_events.parquet"):
            p = h7.OUT / name
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            assert d.index.max() < cutoff if "events" not in name else d["bar_time"].max() < cutoff
            if "events" in name:
                for h in h7.HS:
                    assert d[f"end_{h}"].dropna().max() <= cutoff


def test_nan_rules_gap_and_no_interpolation():
    g0 = pd.Timestamp("2022-01-20 00:00", tz="UTC")
    g1 = g0 + pd.Timedelta("20h")
    perp, spot, fund, m = synth(gap=(g0, g1))
    f = build(perp, spot, fund, m)
    # hours entirely inside the gap have no taker observation
    inside = f.loc[(f.index >= g0 + pd.Timedelta("1h")) & (f.index < g1 - pd.Timedelta("1h"))]
    assert inside["taker_lr"].isna().all()
    assert inside["metrics_gap"].all()
    # as-of value older than 12 h is NaN: OI-based diff NaN deep inside the gap and 1h after it
    late = f.loc[(f.index >= g0 + pd.Timedelta("13h")) & (f.index < g1 - pd.Timedelta("1h"))]
    assert late["doi1_z"].isna().all() and late["ls_top_z"].isna().all()
    # dOI_k is NaN if either endpoint is NaN: the k=24 diff stays NaN for 24 h after an NaN endpoint
    first_nan = late.index[0]
    raw = pd.Series(h7._asof(m["sum_open_interest"], perp.index), index=perp.index)
    d24 = np.log(raw / raw.shift(24))
    assert d24.loc[first_nan].item() != d24.loc[first_nan].item()  # NaN
    # no event is generated from NaN triggers
    ev = h7.make_events(f, perp)
    assert not ev.empty
    assert f.loc[ev["bar_time"][ev.trigger == "taker"], "taker_z24"].notna().all()


def test_taker_hour_needs_six_rows():
    perp, spot, fund, m = synth()
    hr = pd.Timestamp("2022-01-10 05:00", tz="UTC")
    # keep only 4 of 12 rows in that hour (obs in (hr, hr+1h] i.e. create_time in (hr-5min, hr+55min])
    sel = (m.index > hr - pd.Timedelta("5min")) & (m.index <= hr + pd.Timedelta("55min"))
    drop = m.index[sel][4:]
    f = build(perp, spot, fund, m.drop(drop))
    assert np.isnan(f.loc[hr, "taker_lr"])
    assert np.isfinite(f.loc[hr - HOUR, "taker_lr"])


def _flat(n=40):
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    perp = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}, index=idx)
    f = pd.DataFrame(index=idx)
    for c in ("taker_z24", "basis_z7d", "doi4_z", "ret_4h"):
        f[c] = 0.0
    f["liq_long_consumed"] = 0.0
    f["liq_short_consumed"] = 0.0
    f["bar_dir"] = 0.0
    f["atr24"] = 2.0
    return idx, perp, f


def test_events_and_labels_synthetic():
    idx, perp, f = _flat()
    # long taker event at bar 5: entry = open[6]=100, TP 103, SL 98. bar 7 reaches 103.5 -> TP at bar 7
    f.loc[idx[5], "taker_z24"] = 3.0
    perp.loc[idx[7], "high"] = 103.5
    # short basis event at bar 10 (basis z < -2 -> short wait: with the basis => short): SL=102, TP=97
    f.loc[idx[10], "basis_z7d"] = -2.5
    perp.loc[idx[11], ["high", "low"]] = [102.5, 96.0]          # both barriers in one bar -> SL
    # fade OI event at bar 15, ret_4h>0 -> short; price drifts down, TP hit at bar 18
    f.loc[idx[15], ["doi4_z", "ret_4h"]] = [2.5, 0.02]
    perp.loc[idx[18], "low"] = 96.5
    # liq: long cluster consumed at bar 20 -> long; nothing happens -> vertical, exit at close of bar 20+1+h-1
    f.loc[idx[20], "liq_long_consumed"] = 1.0
    perp.loc[idx[22], "close"] = 100.4
    # event on the last bars cannot be labelled
    f.loc[idx[-2], "taker_z24"] = 3.0
    ev = h7.make_events(f, perp)

    def row(trig, bar):
        return ev[(ev.trigger == trig) & (ev.bar_time == idx[bar])].iloc[0]

    r = row("taker", 5)
    assert r.direction == 1 and r.entry_time == idx[6] and r.entry_price == 100.0
    for h in (2, 4, 6):
        assert r[f"y_{h}"] == 1 and r[f"outcome_{h}"] == 1
        assert r[f"end_{h}"] == idx[7] + HOUR
        assert r[f"ret_atr_{h}"] == pytest.approx(1.5)
    r = row("basis", 10)
    assert r.direction == -1 and r.y_2 == 0 and r.outcome_2 == -1 and r.end_2 == idx[11] + HOUR
    assert r.ret_atr_4 == pytest.approx(-1.0)
    r = row("oi", 15)
    assert r.direction == -1 and r.outcome_2 == 0 and r.y_2 == 0       # bars 16,17 quiet -> vertical
    assert r.outcome_4 == 1 and r.end_4 == idx[18] + HOUR and r.outcome_6 == 1
    r = row("liq", 20)
    assert r.direction == 1 and r.outcome_2 == 0 and r.end_2 == idx[22] + HOUR
    assert r.ret_atr_2 == pytest.approx(0.4 / 2.0)
    r = row("taker", len(idx) - 2)
    assert np.isnan(r.y_2) and np.isnan(r.y_4) and np.isnan(r.y_6) and pd.isna(r.end_2)
    # no events elsewhere
    assert len(ev) == 5
    # labels use real prices after the event bar only: changing the event bar's own prices changes nothing
    perp2 = perp.copy()
    perp2.loc[idx[5], ["high", "low", "close"]] = [150.0, 50.0, 120.0]
    ev2 = h7.make_events(f, perp2)
    pd.testing.assert_frame_equal(ev.drop(columns="atr"), ev2.drop(columns="atr"))


def test_liq_summary_causal():
    perp, spot, fund, m = synth(n=600)
    oi = h7._asof(m["sum_open_interest"], perp.index)
    a = h7.atr(perp).to_numpy()
    full = h7.liq_summary(perp.close.values, perp.high.values, perp.low.values, oi, a, warm=50)
    k = 400
    part = h7.liq_summary(perp.close.values[:k], perp.high.values[:k], perp.low.values[:k], oi[:k], a[:k], warm=50)
    pd.testing.assert_frame_equal(full.iloc[:k].reset_index(drop=True), part)
