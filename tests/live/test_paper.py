import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.live import paper

TF, STEP = "4h", 4 * 3600
SPEC = {"name": "t", "tf": TF, "source": "real", "costs": "BINANCE_PERP_TAKER",
        "indicators": {"macross": {"fast": 5, "slow": 20, "kind": "ema"}},
        "long": ["macross.fast_above_slow"], "exit_long": ["macross.fast_below_slow"]}
STRAT = {"id": "t-1", "symbol": "BTCUSDT", "tf": TF, "spec": SPEC}
N = 400
T0 = pd.Timestamp("2026-01-01", tz="UTC")


def make_bars(n=N, seed=1):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.01, n) + 0.05 * np.sin(np.arange(n) / 12) * 0.1))
    open_ = np.r_[close[0], close[:-1]] * (1 + r.normal(0, 0.001, n))
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.003,
                       "low": np.minimum(open_, close) * 0.997, "close": close, "volume": 1.0},
                      index=pd.date_range(T0, periods=n, freq=TF))
    df["funding"] = 0.0
    return df


def con():
    c = sqlite3.connect(":memory:")
    c.executescript(paper.SCHEMA)
    return c


def start_before(bars, k=150):
    """start_ts so that bar k is the first processed bar."""
    return int(bars.index[k].timestamp()) + STEP - 1


def strat(start):
    return {**STRAT, "start_ts": start}


def rows(c, tbl, cols):
    return c.execute(f"SELECT {cols} FROM {tbl} ORDER BY 1,2,3").fetchall()


def run(bars, start, nbars, forming=True):
    c = con()
    b = bars.iloc[:nbars]
    f = (int(bars.index[nbars].timestamp()), float(bars["open"].iloc[nbars])) if forming else None
    paper.process(c, strat(start), b, f, "now")
    return c


def test_no_lookahead_signals_and_fills():
    bars = make_bars()
    start = start_before(bars)
    k = 250
    c1 = run(bars, start, k + 1)
    # wreck every bar after k: ledger rows for bars <= k must not change
    bad = bars.copy()
    bad.iloc[k + 1:, :4] *= 7.0
    c2 = run(bad, start, k + 1)
    cols = "strategy,bar_time,open,high,low,close,long_sig,short_sig,exit_long,exit_short"
    assert rows(c1, "signals", cols) == rows(c2, "signals", cols) and len(rows(c1, "signals", cols)) == k - 150 + 1
    # signals at bar t computed from bars<=t only: compare with the full-history run
    c3 = run(bars, start, N - 1)
    full = {r[1]: r[6:] for r in rows(c3, "signals", cols)}
    for r in rows(c1, "signals", cols):
        assert full[r[1]] == r[6:]


def test_fill_at_next_open_with_costs():
    bars = make_bars()
    start = start_before(bars)
    c = run(bars, start, N - 1)
    fills = c.execute("SELECT trade_no,leg,bar_time,decision_bar_time,side,qty,ref_price,price,fee,reason,"
                      "notional FROM fills ORDER BY trade_no, leg DESC").fetchall()
    assert len(fills) >= 4
    sig = {r[0]: r for r in c.execute("SELECT bar_time,long_sig,exit_long FROM signals")}
    t = {int(ts.timestamp()): i for i, ts in enumerate(bars.index)}
    slip, fee = 2e-4, 5e-4
    for tn, leg, bt, dec, side, qty, ref, px, f, reason, notional in fills:
        if reason in ("signal",) or leg == "entry" or reason == "reverse":
            assert dec == bt - STEP                      # decided at bar t close ...
            assert ref == pytest.approx(bars["open"].iloc[t[bt]])  # ... filled at open of t+1
            assert dec in sig and (sig[dec][1] if leg == "entry" else sig[dec][2]) == 1
        assert px == pytest.approx(ref * (1 + slip) if side == "buy" else ref * (1 - slip))
        assert f == pytest.approx(qty * px * fee) and notional == pytest.approx(qty * px)
    e = [x for x in fills if x[1] == "entry"][0]
    assert e[10] == pytest.approx(paper.INIT_EQUITY)     # first trade: full equity notional
    # closed-trade pnl reconciles with the fills: equity after trade 0 = init + pnl - fees
    ex = [x for x in fills if x[1] == "exit"][0]
    pnl = (ex[7] - e[7]) * e[5] - e[8] - ex[8]
    eq_after = c.execute("SELECT equity FROM equity WHERE bar_time>=? AND position=0 ORDER BY bar_time",
                         (ex[2],)).fetchone()[0]
    assert eq_after == pytest.approx(paper.INIT_EQUITY + pnl, rel=1e-9)


def test_funding_cost():
    bars = make_bars()
    bars["funding"] = 0.0
    bars.iloc[::2, bars.columns.get_loc("funding")] = 0.0001
    start = start_before(bars)
    c = run(bars, start, N - 1)
    f = c.execute("SELECT bar_time,rate,notional,amount FROM funding").fetchall()
    assert f and all(a == pytest.approx(r * n) for _, r, n, a in f)   # long pays rate*notional
    assert all(a > 0 for *_, a in f)


def test_no_backfill_before_start():
    bars = make_bars()
    start = start_before(bars)
    c = run(bars, start, N - 1)
    first = int(bars.index[150].timestamp())
    assert c.execute("SELECT MIN(bar_time) FROM signals").fetchone()[0] == first
    assert c.execute("SELECT MIN(bar_time) FROM equity").fetchone()[0] == first
    assert c.execute("SELECT MIN(bar_time) FROM fills").fetchone()[0] > first
    assert paper.process(con(), strat(int(bars.index[-1].timestamp()) + 10 * STEP), bars) == 0


def snapshot(c):
    out = {}
    for tbl in ("signals", "orders", "fills", "funding", "positions", "equity"):
        cols = [r[1] for r in c.execute(f"PRAGMA table_info({tbl})") if r[1] != "recorded_at"]
        out[tbl] = c.execute(f"SELECT {','.join(cols)} FROM {tbl} ORDER BY 1,2,3").fetchall()
    return out


def test_idempotent_and_incremental_equals_batch():
    bars = make_bars()
    start = start_before(bars)
    batch = run(bars, start, N - 1)
    inc = con()
    for k in range(140, N):                                  # one new closed bar per "minute"
        f = (int(bars.index[k].timestamp()), float(bars["open"].iloc[k]))
        paper.process(inc, strat(start), bars.iloc[:k], f, "now")
        assert paper.process(inc, strat(start), bars.iloc[:k], f, "restart") == 0   # rerun: nothing new
    # appending bar by bar gives exactly the ledger of one batch run (no row ever rewritten)
    assert snapshot(inc) == snapshot(batch)


def test_forming_bar_not_used_for_signals():
    bars = make_bars()
    start = start_before(bars)
    k = 250
    a = run(bars, start, k)
    f_bad = (int(bars.index[k].timestamp()), float(bars["open"].iloc[k]))
    c = con()
    paper.process(c, strat(start), bars.iloc[:k], f_bad, "now")
    assert snapshot(a)["signals"] == snapshot(c)["signals"]
    assert c.execute("SELECT MAX(bar_time) FROM signals").fetchone()[0] == int(bars.index[k - 1].timestamp())
