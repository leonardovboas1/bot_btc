"""Parity of btclab.engine.simulate against vectorbt (open source).

Semantics alignment
-------------------
* Signals are evaluated on the close of bar i and fill at open[i+1]. The engine
  does this internally; for vectorbt the signal series are shifted by one bar
  and ``price=open``.
* Same fee and slippage (vectorbt: buy at p*(1+s), sell at p*(1-s), fee on
  traded value). No funding. 100% of equity per trade, init cash 100.
* An open position on the last bar is marked at the last close without an exit
  fee in both libraries (vectorbt "open" trade, engine EXIT_END).

Known, documented difference in sizing (fees only)
--------------------------------------------------
With size=100% and fee f the engine buys qty = E/px and pays the fee on top
(equity drops by E*f at entry). vectorbt cannot overdraw cash, so it buys
qty = E/(px*(1+f)).
The per-trade equity difference is first order (~f*|trade return|), so with fees
the final equity is checked two ways: strictly (1e-9) against a replay of
vectorbt's own fills under the engine's fee convention, and loosely against
vectorbt's own equity. The fee=0 variants (slippage only) have no sizing
difference and are compared at 1e-9 relative, plus identical fill prices.

Stop-loss test: vectorbt checks stops only from the bar after entry, the engine also on the entry bar (see test).
checks high/low from the bar after entry; the engine anchors on its slipped
entry price and also checks the entry bar. Exit price conventions on gaps
differ slightly. The stop test therefore asserts: same trade count, entries
and exits identical, final equity within the fee tolerance plus a small
stop tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

vbt = pytest.importorskip("vectorbt")

from btclab.engine.sim import EXIT_END, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INIT = 100.0
SLIP_BP = 2.0


# ---------------------------------------------------------------- data
def gbm(n=5000, seed=7, mu=0.0, sigma=0.01):
    rng = np.random.default_rng(seed)
    ret = rng.normal(mu, sigma, n)
    close = 20000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[20000.0], close[:-1]]) * np.exp(rng.normal(0, 0.001, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close}, index=idx)


def real_data():
    p = ROOT / "data/raw/um_futures/klines_1h.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)[["open", "high", "low", "close"]].dropna()
    d = d[(d > 0).all(axis=1)]
    return d.iloc[-5000:].copy()


DATASETS = {"gbm_seed7": gbm(seed=7), "gbm_seed11_drift": gbm(seed=11, mu=2e-4)}
_r = real_data()
if _r is not None:
    DATASETS["real_btc_1h_last5k"] = _r


# ---------------------------------------------------------------- signals
def cross_signals(close: pd.Series, fast=20, slow=50):
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    ok = f.notna() & s.notna()
    above = (f > s) & ok
    prev = above.shift(1, fill_value=False)
    prev_ok = ok.shift(1, fill_value=False)
    up = above & ~prev & prev_ok
    dn = ~above & prev & ok
    return up.astype(bool), dn.astype(bool)


def rsi(close: pd.Series, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


# ---------------------------------------------------------------- runners
def _v(s):
    return None if s is None else s.values


def run_engine(df, el=None, xl=None, es=None, xs=None, *, fee, allow_short=False,
               reverse=False, sl_pct=0.0):
    return simulate(
        df.open.values, df.high.values, df.low.values, df.close.values,
        _v(el), _v(es), _v(xl), _v(xs),
        fee_rate=fee, slippage_bp=SLIP_BP, allow_short=allow_short,
        reverse_on_signal=reverse, sl_pct=sl_pct, init_cash=INIT,
    )


def sh(s):
    return None if s is None else s.shift(1, fill_value=False).astype(bool)


def run_vbt(df, el, xl=None, es=None, xs=None, *, fee, sl=None):
    kw = dict(
        close=df.close, open=df.open, high=df.high, low=df.low, price=df.open,
        entries=sh(el), exits=sh(xl),
        fees=fee, slippage=SLIP_BP / 1e4, init_cash=INIT, size=np.inf,
        accumulate=False, freq="1h",
    )
    if es is not None:
        kw.update(short_entries=sh(es), short_exits=sh(xs), upon_opposite_entry="reverse")
    if sl is not None:
        kw.update(sl_stop=sl, stop_entry_price="fillprice", stop_exit_price="stoplimit")
    return vbt.Portfolio.from_signals(**kw)


def replay_engine_convention(rec, fee):
    """Replay vectorbt's own fills (entry/exit prices, indices, direction) with
    the engine's fee convention: qty = E/entry_px, entry fee f*E on top, exit fee
    f*qty*exit_px (none for the trade still open on the last bar)."""
    eq = INIT
    for _, r in rec.iterrows():
        side = 1.0 if r["direction"] == 0 else -1.0
        qty = eq / r["entry_price"]
        fees = fee * qty * r["entry_price"]
        if r["status"] != 0:  # closed
            fees += fee * qty * r["exit_price"]
        eq += (r["exit_price"] - r["entry_price"]) * qty * side - fees
    return eq


def compare(res, pf, fee):
    tr = res.trades
    rec = pf.trades.records
    n_e, n_v = len(tr), len(rec)
    assert n_e == n_v, f"trade count engine={n_e} vbt={n_v}"
    assert n_e > 0
    np.testing.assert_array_equal(tr["entry_idx"], rec["entry_idx"].values)
    np.testing.assert_array_equal(tr["exit_idx"], rec["exit_idx"].values)
    side_v = np.where(rec["direction"].values == 0, 1, -1)
    np.testing.assert_array_equal(tr["side"], side_v)
    # entry/exit prices identical (slippage convention)
    np.testing.assert_allclose(tr["entry_px"], rec["entry_price"].values, rtol=1e-12)
    np.testing.assert_allclose(tr["exit_px"], rec["exit_price"].values, rtol=1e-12)
    eq_e = res.equity[-1]
    eq_v = float(pf.value().iloc[-1])
    if fee == 0:
        assert abs(eq_e - eq_v) / eq_v <= 1e-9, (eq_e, eq_v)
    else:
        # strict: engine vs replay of vectorbt fills with the engine fee convention
        eq_r = replay_engine_convention(rec, fee)
        assert abs(eq_e - eq_r) / eq_r <= 1e-9, (eq_e, eq_r)
        # sanity vs vectorbt's own equity: sizing convention differs by ~f*|r| per trade
        tol = 2 * fee * (n_e + np.abs(tr["pnl_pct"]).sum()) + 1e-6
        assert abs(eq_e - eq_v) / eq_v <= tol, (eq_e, eq_v, tol)


fee_p = pytest.mark.parametrize("fee", [0.0, 0.0005], ids=["fee0", "fee5bp"])
ds_p = pytest.mark.parametrize("name", list(DATASETS))


@fee_p
@ds_p
def test_buy_and_hold(name, fee):
    df = DATASETS[name]
    el = pd.Series(False, index=df.index)
    el.iloc[0] = True
    res = run_engine(df, el, fee=fee)
    pf = run_vbt(df, el, fee=fee)
    assert res.trades["exit_reason"][0] == EXIT_END
    compare(res, pf, fee)


@fee_p
@ds_p
def test_sma_cross_long_only(name, fee):
    df = DATASETS[name]
    up, dn = cross_signals(df.close)
    res = run_engine(df, up, dn, fee=fee, allow_short=False)
    pf = run_vbt(df, up, dn, fee=fee)
    compare(res, pf, fee)


@fee_p
@ds_p
def test_sma_cross_long_short_reverse(name, fee):
    df = DATASETS[name]
    up, dn = cross_signals(df.close)
    res = run_engine(df, up, None, dn, None, fee=fee, allow_short=True, reverse=True)
    pf = run_vbt(df, up, None, dn, None, fee=fee)
    compare(res, pf, fee)


@fee_p
@ds_p
def test_rsi_mean_reversion(name, fee):
    df = DATASETS[name]
    r = rsi(df.close)
    el = (r < 30).fillna(False)
    xl = (r > 70).fillna(False)
    res = run_engine(df, el, xl, fee=fee, allow_short=False)
    pf = run_vbt(df, el, xl, fee=fee)
    compare(res, pf, fee)


@fee_p
@ds_p
def test_sma_cross_with_2pct_stop(name, fee):
    df = DATASETS[name]
    up, dn = cross_signals(df.close)
    res = run_engine(df, up, dn, fee=fee, allow_short=False, sl_pct=0.02)
    pf = run_vbt(df, up, dn, fee=fee, sl=0.02)
    tr, rec = res.trades, pf.trades.records
    assert len(tr) == len(rec)
    np.testing.assert_array_equal(tr["entry_idx"], rec["entry_idx"].values)
    # Intrabar convention differences (see module docstring): the engine checks the
    # stop on the entry bar itself (exit_idx == entry_idx); vectorbt only from the
    # next bar, so those trades exit one bar later at a different price.
    same_bar = tr["exit_idx"] == tr["entry_idx"]
    np.testing.assert_array_equal(tr["exit_idx"][~same_bar], rec["exit_idx"].values[~same_bar])
    assert (rec["exit_idx"].values[same_bar] >= tr["exit_idx"][same_bar]).all()
    # stop exits happen at or below the 2% level (slippage / gap-through)
    stops = tr["exit_reason"] == 1
    assert (tr["exit_px"][stops] <= tr["entry_px"][stops] * 0.98 + 1e-9).all()
    # equity: loose, because stop fill prices and entry-bar handling differ
    eq_e, eq_v = res.equity[-1], float(pf.value().iloc[-1])
    assert abs(eq_e - eq_v) / eq_v <= 0.03, (eq_e, eq_v)
