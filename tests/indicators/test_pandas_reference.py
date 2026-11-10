"""Compare against independent pandas reference implementations."""
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from btclab.indicators import base as ta
from btclab.indicators import cipher_b, momentum, stoch_rsi, trend

RTOL = 1e-10


def ewm_seeded(x, n, alpha):
    """Pine-style exponential filter via pandas: SMA seed at the first full window of
    valid values, then ewm(adjust=False)."""
    s = pd.Series(np.asarray(x, dtype=float))
    out = pd.Series(np.nan, index=s.index)
    first = s.first_valid_index()
    if first is None:
        return out.to_numpy()
    seed_at = first + n - 1
    if seed_at >= len(s):
        return out.to_numpy()
    tail = s.iloc[seed_at:].copy()
    tail.iloc[0] = s.iloc[first:seed_at + 1].mean()
    out.iloc[seed_at:] = tail.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return out.to_numpy()


def ema_ref(x, n):
    return ewm_seeded(x, n, 2 / (n + 1))


def rma_ref(x, n):
    return ewm_seeded(x, n, 1 / n)


def close_(a, b, rtol=RTOL, atol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN masks differ"
    m = ~np.isnan(a)
    assert_allclose(a[m], b[m], rtol=rtol, atol=atol)


def test_sma_wma_highest_lowest(ohlcv):
    c = ohlcv["close"]
    close_(ta.sma(c, 20), c.rolling(20).mean())
    w = np.arange(1, 15, dtype=float)
    close_(ta.wma(c, 14), c.rolling(14).apply(lambda v: np.dot(v, w) / w.sum(), raw=True))
    close_(ta.highest(ohlcv["high"], 10), ohlcv["high"].rolling(10).max())
    close_(ta.lowest(ohlcv["low"], 10), ohlcv["low"].rolling(10).min())


def test_ema_rma(ohlcv):
    c = ohlcv["close"].to_numpy()
    for n in (3, 9, 26, 200):
        close_(ta.ema(c, n), ema_ref(c, n))
        close_(ta.rma(c, n), rma_ref(c, n))


def test_rsi_atr(ohlcv):
    c = ohlcv["close"]
    d = c.diff()
    up = rma_ref(d.clip(lower=0), 14)
    dn = rma_ref((-d).clip(lower=0), 14)
    ref = 100 - 100 / (1 + up / dn)
    close_(ta.rsi(c, 14), ref)
    h, l = ohlcv["high"], ohlcv["low"]
    trv = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    trv.iloc[0] = h.iloc[0] - l.iloc[0]
    close_(ta.atr(h, l, c, 14), rma_ref(trv, 14))


def test_stoch_cci_willr_mfi(ohlcv):
    h, l, c, v = ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    close_(ta.stoch(c, h, l, 14), 100 * (c - ll) / (hh - ll))
    close_(momentum.williams_r(h, l, c, 14)["wr"], 100 * (c - hh) / (hh - ll))
    tp = (h + l + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    close_(momentum.cci(h, l, c, 20)["cci"], (tp - tp.rolling(20).mean()) / (0.015 * mad), rtol=1e-8)
    ch = tp.diff()
    pos = (v * tp.where(~(ch <= 0), 0.0)).rolling(14).sum()
    neg = (v * tp.where(~(ch >= 0), 0.0)).rolling(14).sum()
    close_(momentum.mfi(h, l, c, v, 14)["mfi"], 100 - 100 / (1 + pos / neg), rtol=1e-8)


def test_macd(ohlcv):
    c = ohlcv["close"].to_numpy()
    m = ema_ref(c, 12) - ema_ref(c, 26)
    s = ema_ref(m, 9)
    out = momentum.macd(c)
    close_(out["macd"], m)
    close_(out["signal"], s)


def test_stoch_rsi(ohlcv):
    c = ohlcv["close"]
    r = pd.Series(ta.rsi(c, 14))
    st = 100 * (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min())
    k = st.rolling(3).mean()
    d = k.rolling(3).mean()
    out = stoch_rsi.stoch_rsi(c)
    close_(out["k"], k, rtol=1e-8)
    close_(out["d"], d, rtol=1e-8)


def test_wavetrend(ohlcv):
    ap = ((ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3).to_numpy()
    esa = ema_ref(ap, 9)
    de = ema_ref(np.abs(ap - esa), 9)
    ci = (ap - esa) / (0.015 * de)
    wt1 = ema_ref(ci, 12)
    wt2 = pd.Series(wt1).rolling(3).mean().to_numpy()
    a1, a2, vw = cipher_b.wavetrend(ap, 9, 12, 3)
    close_(a1, wt1, rtol=1e-8)
    close_(a2, wt2, rtol=1e-8)


def test_dmi_reference(ohlcv):
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    up, dn = h.diff(), -l.diff()
    pdm = np.where(up.isna(), np.nan, np.where((up > dn) & (up > 0), up, 0.0))
    mdm = np.where(dn.isna(), np.nan, np.where((dn > up) & (dn > 0), dn, 0.0))
    trv = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    trv.iloc[0] = np.nan
    trur = rma_ref(trv, 14)
    plus = pd.Series(100 * rma_ref(pdm, 14) / trur).ffill().to_numpy()
    minus = pd.Series(100 * rma_ref(mdm, 14) / trur).ffill().to_numpy()
    s = plus + minus
    adx = 100 * rma_ref(np.abs(plus - minus) / np.where(s == 0, 1, s), 14)
    out = trend.dmi(h, l, c)
    close_(out["plus_di"], plus, rtol=1e-8)
    close_(out["adx"], adx, rtol=1e-8)


def test_supertrend_reference(ohlcv):
    """Straightforward pandas-loop transcription of the Pine ta.supertrend pseudo-code."""
    h, l, c = (ohlcv[k].to_numpy() for k in ("high", "low", "close"))
    a = ta.atr(h, l, c, 10)
    src = (h + l) / 2
    U, L, st, dr = [], [], [], []
    for i in range(len(c)):
        ub, lb = src[i] + 3 * a[i], src[i] - 3 * a[i]
        pl = 0.0 if i == 0 or np.isnan(L[-1]) else L[-1]
        pu = 0.0 if i == 0 or np.isnan(U[-1]) else U[-1]
        c1 = c[i - 1] if i else np.nan
        lb = lb if (lb > pl or c1 < pl) else pl
        ub = ub if (ub < pu or c1 > pu) else pu
        if i == 0 or np.isnan(a[i - 1]):
            d = 1
        elif st[-1] == pu:
            d = -1 if c[i] > ub else 1
        else:
            d = 1 if c[i] < lb else -1
        U.append(ub); L.append(lb); dr.append(d); st.append(lb if d == -1 else ub)
    st = np.where(np.isnan(a), np.nan, st)
    out = trend.supertrend(h, l, c, 10, 3.0)
    close_(out["supertrend"], st)
