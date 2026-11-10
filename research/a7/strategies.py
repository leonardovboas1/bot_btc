"""Amendment 7 strategy classes C1/C2/C3 and benchmarks. Targets at row t use data <= close t only."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btclab.api.presets import get_preset  # noqa: E402
from btclab.data.loaders import get_cutoff, load_funding, load_ohlcv  # noqa: E402
from btclab.engine.metrics import daily_returns  # noqa: E402
from btclab.signals.combiner import run_spec  # noqa: E402
from research.a7.portfolio import PERP_COST, SPOT_COST, Result, backtest  # noqa: E402

DESIGN = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "NEARUSDT", "FETUSDT"]
DESIGN_END = get_cutoff().floor("D")          # exclusive: last full day before the cutoff bar
V2_START = DESIGN_END + pd.Timedelta(days=1)  # first full day after the cutoff day
V3_START, V3_END = pd.Timestamp("2017-08-17", tz="UTC"), pd.Timestamp("2019-09-09", tz="UTC")  # end exclusive
PPY = 365


@dataclass
class Panel:
    close: pd.DataFrame            # daily close of the traded instrument (perp, or spot for V3)
    ret: pd.DataFrame
    fund: pd.DataFrame | None      # daily summed funding (perp) or None
    fee: float
    spot_close: pd.DataFrame | None = None
    kind: str = "perp"
    no_spot: tuple = ()            # symbols whose spot hedge leg fell back to the perp price


def load_panel(symbols, kind="perp", start=None, end=None, allow_holdout=False, need_spot=False) -> Panel:
    cl, fu, sp, nos = {}, {}, {}, []
    for s in symbols:
        try:
            d = load_ohlcv(kind, "1d", start, end, allow_holdout, s)
        except FileNotFoundError:
            continue
        if len(d) < 30:
            continue
        cl[s] = d["close"]
        if kind == "perp":
            fu[s] = load_funding("1d", start, end, allow_holdout, s)
        if need_spot:
            try:
                sp[s] = load_ohlcv("spot", "1d", start, end, allow_holdout, s)["close"]
            except FileNotFoundError:
                sp[s] = d["close"]       # no spot file: hedge leg tracks perp (basis ignored)
                nos.append(s)
    close = pd.DataFrame(cl).sort_index()
    close = close.reindex(pd.date_range(close.index[0], close.index[-1], freq="D", tz="UTC"))
    fund = pd.DataFrame(fu).reindex(close.index) if kind == "perp" else None
    spot = pd.DataFrame(sp).reindex(close.index) if need_spot else None
    return Panel(close, close.pct_change(fill_method=None), fund, PERP_COST if kind == "perp" else SPOT_COST,
                 spot, kind, tuple(nos))


# ---------------------------------------------------------------- C1
def _ewma_vol(close: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(close).diff()
    return lr.ewm(span=60, min_periods=60).std() * np.sqrt(PPY)


def c1_targets(close: pd.DataFrame, long_only: bool, sigma_t: float) -> pd.DataFrame:
    """s = mean sign of 20/60/120d log returns; w = s * min(1, sigma_t / sigma_60); gross cap 2 in backtest."""
    lc = np.log(close)
    s = sum(np.sign(lc - lc.shift(n)) for n in (20, 60, 120)) / 3.0
    if long_only:
        s = s.clip(lower=0)
    pos = s * (sigma_t / _ewma_vol(close)).clip(upper=1.0)
    return pos.fillna(0.0)


def run_c1(p: Panel, long_only: bool, sigma_t: float) -> Result:
    return backtest(p.ret, c1_targets(p.close, long_only, sigma_t), p.fee, p.fund, gross_cap=2.0)


# ---------------------------------------------------------------- C2
def c2_targets(close: pd.DataFrame, L: int, long_short: bool) -> pd.DataFrame:
    mom = close.shift(1) / close.shift(1 + L) - 1          # L-day return skipping the last day
    vol = _ewma_vol(close)
    M, V = mom.to_numpy(), vol.to_numpy()
    out = np.full(M.shape, np.nan)
    for t in np.where(close.index.dayofweek == 6)[0]:      # Sunday close == Monday 00:00 UTC
        ok = np.isfinite(M[t]) & np.isfinite(V[t])
        out[t] = 0.0
        m = int(ok.sum())
        if m == 0 or (long_short and m < 2):
            continue
        n = max(1, round(m / 3))
        if long_short:
            n = min(n, m // 2)
        order = np.where(ok)[0][np.argsort(-M[t][ok])]
        top, bot = order[:n], order[::-1][:n]
        wt = 1.0 / V[t][top]
        out[t][top] = (0.5 if long_short else 1.0) * wt / wt.sum()
        if long_short:
            wb = 1.0 / V[t][bot]
            out[t][bot] = -0.5 * wb / wb.sum()
    return pd.DataFrame(out, close.index, close.columns)


def run_c2(p: Panel, L: int, long_short: bool) -> Result:
    return backtest(p.ret, c2_targets(p.close, L, long_short), p.fee, p.fund, gross_cap=2.0)


# ---------------------------------------------------------------- C3
def run_c3(p: Panel, theta: float) -> Result:
    """Long spot + short perp (each 0.5/k of equity, k = open carries). Open when the 7d mean funding
    annualised > theta, close when it is < 0 (decided at close t, active t+1)."""
    assert p.fund is not None and p.spot_close is not None
    perp_r = p.ret
    spot_r = p.spot_close.pct_change(fill_method=None)
    m = p.fund.rolling(7, min_periods=7).mean() * PPY
    both = perp_r.notna() & spot_r.notna()
    on = np.zeros(m.shape, bool)
    M = m.to_numpy()
    state = np.zeros(m.shape[1], bool)
    for t in range(len(m)):
        x = M[t]
        state = np.where(state, ~(x < 0), x > theta) & np.isfinite(x)
        on[t] = state
    on &= both.to_numpy()
    for j, c in enumerate(p.close.columns):      # no spot pair -> cannot carry
        if c in p.no_spot:
            on[:, j] = False
    k = on.sum(axis=1, keepdims=True)
    w = np.where(on, 0.5 / np.maximum(k, 1), 0.0)
    cols = list(p.close.columns)
    ix = p.close.index
    ret = pd.concat([spot_r.add_suffix("|spot"), perp_r.add_suffix("|perp")], axis=1)
    tgt = pd.concat([pd.DataFrame(w, ix, cols).add_suffix("|spot"),
                     pd.DataFrame(-w, ix, cols).add_suffix("|perp")], axis=1)
    fund = pd.concat([pd.DataFrame(0.0, ix, cols).add_suffix("|spot"), p.fund.add_suffix("|perp")], axis=1)
    fee = np.r_[np.full(len(cols), SPOT_COST), np.full(len(cols), PERP_COST)]
    return backtest(ret, tgt, fee, fund, gross_cap=1.0)


# ---------------------------------------------------------------- benchmarks
def bench_ew(p: Panel) -> Result:
    a = p.close.notna().astype(float)
    return backtest(p.ret, a.div(a.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0), p.fee, p.fund)


def bench_btc(p: Panel) -> Result:
    col = "BTCUSDT"
    t = pd.DataFrame(1.0, p.close.index, [col])
    return backtest(p.ret[[col]], t, p.fee, None if p.fund is None else p.fund[[col]])


def bench_brayvb(symbols, kind="perp", start=None, end=None, allow_holdout=False) -> pd.Series:
    """brayvb_v2 default 4h-HA per coin (own capital), equal-weight average of the daily returns of the
    coins that exist that day. Costs: perp -> BINANCE_PERP_TAKER + funding; spot -> SPOT, long-only."""
    rs = {}
    for s in symbols:
        try:
            df = load_ohlcv(kind, "4h", start, end, allow_holdout, s)
        except FileNotFoundError:
            continue
        if len(df) < 200:
            continue
        spec = get_preset("brayvb_v2")
        spec["source"] = "ha"
        if kind == "perp":
            f = load_funding("4h", start, end, allow_holdout, s).reindex(df.index).fillna(0.0)
            df = df.assign(funding=f.to_numpy())
        else:
            spec["costs"] = "SPOT"
        res, _ = run_spec(spec, df)
        r = daily_returns(np.asarray(res.equity, float), df.index, init_equity=1.0)
        r.index = r.index.tz_localize("UTC")
        rs[s] = r
    d = pd.DataFrame(rs).sort_index()
    return d.mean(axis=1, skipna=True).rename("brayvb")


# ---------------------------------------------------------------- registry
def configs():
    """The 12 registered configs: name -> callable(Panel) -> Result."""
    c = {}
    for lo in (True, False):
        for sg in (0.20, 0.40):
            c[f"C1_{'LO' if lo else 'LS'}_s{int(sg * 100)}"] = (lambda p, lo=lo, sg=sg: run_c1(p, lo, sg))
    for L in (14, 28, 56):
        for ls in (False, True):
            c[f"C2_{'LS' if ls else 'LO'}_L{L}"] = (lambda p, L=L, ls=ls: run_c2(p, L, ls))
    for th in (0.10, 0.20):
        c[f"C3_th{int(th * 100)}"] = (lambda p, th=th: run_c3(p, th))
    return c
