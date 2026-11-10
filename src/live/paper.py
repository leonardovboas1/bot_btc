"""Forward paper trading service (append-only SQLite ledger, closed bars only).

Every minute, for each tracked strategy: fetch closed Binance USDT-perp candles (public REST),
compute the signals with the existing engine pieces on CLOSED bars only, simulate fills at the
next bar open (BINANCE_PERP_TAKER costs + funding) and append the result to
``data/paper/paper.sqlite``.

Design
* The ledger starts when a strategy is first registered (``start_ts``). Signals of bars that
  closed before that are masked out (flat start); history is only used for indicator warm-up.
* Fills/positions/equity are produced by the engine ``simulate`` over the full history (signals
  masked before the start), with the forming bar appended as an open-only row so the fill at its
  open is known immediately. Signals themselves never see that row.
* Every ledger row is keyed (strategy + bar/trade ordinal) and written with INSERT OR IGNORE, so
  restarts and re-computation are idempotent and past rows are never rewritten.
Run:  python -m src.live.paper [--once]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btclab.api.presets import get_preset  # noqa: E402
from btclab.data import loaders  # noqa: E402
from btclab.engine.sim import EXIT_REASON_NAMES, simulate  # noqa: E402
from btclab.indicators import base as _base  # noqa: E402
from btclab.signals.combiner import ATR_LEN, _cached_compute, _cost_model, heikin_ashi  # noqa: E402
from btclab.signals.dsl import RISK_KEYS, build_signal, validate_spec  # noqa: E402

PAPER_DIR = ROOT / "data" / "paper"
DB_PATH = PAPER_DIR / "paper.sqlite"
CONFIG_PATH = PAPER_DIR / "strategies.json"
INIT_EQUITY = 10000.0
FAPI = "https://fapi.binance.com"
CYCLE_SECONDS = 60
STOP_REASONS = (1, 2, 3)      # exits that fill intrabar, after the funding charge of their bar
DEFAULT_CONFIG = {"strategies": [
    {"preset": "brayvb_v2", "symbol": "BTCUSDT", "tf": tf, "source": src}
    for tf in ("1d", "4h") for src in ("real", "ha")]}

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies(id TEXT PRIMARY KEY, symbol TEXT, tf TEXT, source TEXT,
  spec TEXT, start_ts INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS signals(strategy TEXT, bar_time INTEGER, bar_close_time INTEGER,
  open REAL, high REAL, low REAL, close REAL, long_sig INTEGER, short_sig INTEGER,
  exit_long INTEGER, exit_short INTEGER, recorded_at TEXT, PRIMARY KEY(strategy, bar_time));
CREATE TABLE IF NOT EXISTS orders(strategy TEXT, trade_no INTEGER, leg TEXT, decision_bar_time INTEGER,
  fill_bar_time INTEGER, kind TEXT, action TEXT, reason TEXT, recorded_at TEXT,
  PRIMARY KEY(strategy, trade_no, leg));
CREATE TABLE IF NOT EXISTS fills(strategy TEXT, trade_no INTEGER, leg TEXT, bar_time INTEGER,
  decision_bar_time INTEGER, side TEXT, position_side INTEGER, qty REAL, ref_price REAL,
  price REAL, notional REAL, fee REAL, slippage_bp REAL, reason TEXT, recorded_at TEXT,
  PRIMARY KEY(strategy, trade_no, leg));
CREATE TABLE IF NOT EXISTS funding(strategy TEXT, bar_time INTEGER, rate REAL, notional REAL,
  amount REAL, recorded_at TEXT, PRIMARY KEY(strategy, bar_time));
CREATE TABLE IF NOT EXISTS positions(strategy TEXT, bar_time INTEGER, position INTEGER, qty REAL,
  entry_px REAL, mark_px REAL, unrealized REAL, recorded_at TEXT, PRIMARY KEY(strategy, bar_time));
CREATE TABLE IF NOT EXISTS equity(strategy TEXT, bar_time INTEGER, equity REAL, price REAL,
  position INTEGER, recorded_at TEXT, PRIMARY KEY(strategy, bar_time));
CREATE TABLE IF NOT EXISTS runs(ts INTEGER PRIMARY KEY, started_at TEXT, duration_s REAL,
  n_strategies INTEGER, n_new_rows INTEGER, errors TEXT, last_bars TEXT);
"""


# ------------------------------------------------------------------ helpers
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def tf_seconds(tf: str) -> int:
    return int(pd.Timedelta(tf).total_seconds())


def strategy_id(c: dict) -> str:
    return c.get("id") or f"{c.get('preset', 'spec')}-{c['symbol']}-{c['tf']}-{c.get('source', 'real')}"


def load_config(path: Path = CONFIG_PATH) -> list[dict]:
    """Config JSON: {"strategies": [{"preset"|"spec", "symbol", "tf", "source", "id"?}]}."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    out = []
    for c in json.loads(path.read_text())["strategies"]:
        c = dict(c)
        c.setdefault("symbol", "BTCUSDT")
        c.setdefault("source", "real")
        spec = c.get("spec") or get_preset(c["preset"])
        c["id"] = strategy_id(c)
        c["spec"] = {**spec, "tf": c["tf"], "source": c["source"]}
        out.append(c)
    return out


def align_funding(index: pd.DatetimeIndex, events: pd.Series, tf: str) -> np.ndarray:
    """Funding rates summed onto the bar containing each event (same rule as loaders.load_funding)."""
    out = np.zeros(len(index))
    if len(index) and len(events):
        step = pd.Timedelta(tf).to_timedelta64()
        pos = index.searchsorted(events.index, side="right") - 1
        ok = pos >= 0
        pos_c = np.clip(pos, 0, len(index) - 1)
        ok &= events.index.values < (index.values[pos_c] + step)
        np.add.at(out, pos[ok], events.values[ok])
    return out


# ------------------------------------------------------------------ core (no network)
def signals_closed(spec: dict, df: pd.DataFrame):
    """Signal arrays on CLOSED bars (same construction as combiner.run_spec). Returns (sigs, atr)."""
    validate_spec(spec)
    n = len(df)
    inds = spec["indicators"]
    source = spec.get("source", "real")
    sig_df = heikin_ashi(df) if source == "ha" else df
    cache: dict = {}

    def lookup(prim):
        name, _, key = prim.partition(".")
        return _cached_compute(cache, sig_df, name, inds[name], source)[key]

    sigs = {s: build_signal(spec.get(s), lookup, n) for s in ("long", "short", "exit_long", "exit_short")}
    col = lambda c: df[c].to_numpy(dtype=np.float64)  # noqa: E731
    return sigs, _base.atr(col("high"), col("low"), col("close"), ATR_LEN)


def process(con: sqlite3.Connection, strat: dict, bars: pd.DataFrame,
            forming: tuple | None = None, now_iso: str | None = None) -> int:
    """Append new ledger rows for one strategy. ``bars``: CLOSED bars (UTC index, OHLCV and an
    optional ``funding`` column, full history). ``forming``: (open_time_s, open_price) of the bar
    after the last closed one, used only as a fill price. Returns the number of rows inserted."""
    sid, spec, tf = strat["id"], strat["spec"], strat["tf"]
    step = tf_seconds(tf)
    now_iso = now_iso or _now_iso()
    row = con.execute("SELECT start_ts FROM strategies WHERE id=?", (sid,)).fetchone()
    if row is None:
        con.execute("INSERT INTO strategies VALUES(?,?,?,?,?,?,?)",
                    (sid, strat["symbol"], tf, spec.get("source", "real"), json.dumps(spec),
                     strat.get("start_ts") or int(time.time()), now_iso))
        con.commit()
        row = con.execute("SELECT start_ts FROM strategies WHERE id=?", (sid,)).fetchone()
    start_ts = row[0]
    n = len(bars)
    if n == 0:
        return 0
    t = (bars.index.asi8 // 10**9).astype(np.int64)
    first = int(np.searchsorted(t + step, start_ts, side="right"))  # first bar closing after start
    if first >= n:
        return 0
    if forming is not None and int(forming[0]) != int(t[-1]) + step:
        forming = None                                             # not the next bar: ignore

    sigs, atr = signals_closed(spec, bars)
    zeros = np.zeros(n, bool)

    def prep(a):
        a = zeros.copy() if a is None else np.asarray(a, bool).copy()
        a[:first] = False                                          # nothing before the start
        return a

    flags = {k: prep(v) for k, v in sigs.items()}
    col = lambda c: bars[c].to_numpy(dtype=np.float64)  # noqa: E731
    o, h, lo, c = col("open"), col("high"), col("low"), col("close")
    fund = col("funding") if "funding" in bars.columns else np.zeros(n)
    m = n
    t_ext = t
    if forming is not None:                                         # open-only row, no signals on it
        fo = float(forming[1])
        o, h, lo, c = (np.append(x, fo) for x in (o, h, lo, c))
        fund, atr = np.append(fund, 0.0), np.append(atr, np.nan)
        flags = {k: np.append(v, False) for k, v in flags.items()}
        t_ext = np.append(t, t[-1] + step)
        m = n + 1
    cm = _cost_model(spec.get("costs", "BINANCE_PERP_TAKER"))
    kw = cm.sim_kwargs()
    if sigs["short"] is None:
        kw["allow_short"] = False
    risk = {k: v for k, v in (spec.get("risk") or {}).items() if k in RISK_KEYS and v}
    res = simulate(o, h, lo, c, flags["long"], flags["short"], flags["exit_long"], flags["exit_short"],
                   atr=atr, funding=fund, init_cash=INIT_EQUITY, **kw, **risk)
    slip = kw["slippage_bp"] / 1e4
    fee_rate = kw["fee_rate"]
    before = con.total_changes

    def ins(tbl, vals):
        con.execute(f"INSERT OR IGNORE INTO {tbl} VALUES({','.join('?' * len(vals))})", vals)

    # signals: one row per processed closed bar (the exact bar used is stored with it)
    for i in range(first, n):
        ins("signals", (sid, int(t[i]), int(t[i]) + step, o[i], h[i], lo[i], c[i], int(flags["long"][i]),
                        int(flags["short"][i]), int(flags["exit_long"][i]), int(flags["exit_short"][i]),
                        now_iso))

    # fills / orders / funding from the engine trade list
    pos_qty = np.zeros(m)
    pos_entry = np.zeros(m)
    cash = INIT_EQUITY
    fund_rows: dict[int, tuple] = {}
    for k, tr in enumerate(res.trades):
        ei, xi, side, reason = int(tr["entry_idx"]), int(tr["exit_idx"]), int(tr["side"]), int(tr["exit_reason"])
        epx, xpx = float(tr["entry_px"]), float(tr["exit_px"])
        qty = cash * res.params.size_frac * res.params.leverage / epx
        open_end = reason == 6
        dec_in = int(t_ext[ei - 1])
        eref = epx / (1 + slip) if side == 1 else epx / (1 - slip)
        ins("orders", (sid, k, "entry", dec_in, int(t_ext[ei]), "market_next_open",
                       "enter_long" if side == 1 else "enter_short", "signal", now_iso))
        ins("fills", (sid, k, "entry", int(t_ext[ei]), dec_in, "buy" if side == 1 else "sell", side, qty,
                      eref, epx, qty * epx, qty * epx * fee_rate, slip * 1e4, "signal", now_iso))
        pos_qty[ei:xi + 1] = qty
        pos_entry[ei:xi + 1] = epx
        last_fund = xi
        if not open_end:
            intrabar = reason in STOP_REASONS
            dec_out = int(t_ext[xi]) if intrabar else int(t_ext[xi - 1])
            xref = xpx / (1 - slip) if side == 1 else xpx / (1 + slip)
            nm = EXIT_REASON_NAMES.get(reason, str(reason))
            ins("orders", (sid, k, "exit", dec_out, int(t_ext[xi]), "stop" if intrabar else "market_next_open",
                           "exit_long" if side == 1 else "exit_short", nm, now_iso))
            ins("fills", (sid, k, "exit", int(t_ext[xi]), dec_out, "sell" if side == 1 else "buy", side, qty,
                          xref, xpx, qty * xpx, qty * xpx * fee_rate, slip * 1e4, nm, now_iso))
            cash += float(tr["pnl"])
            pos_qty[xi] = qty if intrabar else 0.0   # position at the close of the exit bar is flat
            pos_entry[xi] = epx if intrabar else 0.0
            last_fund = xi if intrabar else xi - 1
        for i in range(ei, min(last_fund, n - 1) + 1):               # funding is charged at the bar open
            if fund[i] != 0.0:
                fund_rows[i] = (float(fund[i]), qty * o[i], float(fund[i]) * qty * o[i] * side)
    for i, (rate, notional, amt) in fund_rows.items():
        if i >= first:
            ins("funding", (sid, int(t[i]), rate, notional, amt, now_iso))

    # position and equity snapshots at every processed closed bar
    for i in range(first, n):
        p = int(res.position[i])
        q = float(pos_qty[i]) if p else 0.0
        e = float(pos_entry[i]) if p else 0.0
        ins("positions", (sid, int(t[i]), p, q, e, c[i], (c[i] - e) * q * p if p else 0.0, now_iso))
        ins("equity", (sid, int(t[i]), float(res.equity[i]), c[i], p, now_iso))
    con.commit()
    return con.total_changes - before


# ------------------------------------------------------------------ market data
class Market:
    """History per (symbol, tf): on-disk parquet + REST tail, closed bars only."""

    def __init__(self, session=None):
        self.s = session or requests.Session()
        self.hist: dict = {}
        self.fund: dict = {}
        self.server_ms = 0

    def get(self, path, **params):
        r = self.s.get(FAPI + path, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def sync_time(self):
        self.server_ms = int(self.get("/fapi/v1/time")["serverTime"])

    @staticmethod
    def _rows(rows) -> pd.DataFrame:
        df = pd.DataFrame([[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[7]),
                            float(x[8]), float(x[9]), float(x[10])] for x in rows],
                          columns=["open", "high", "low", "close", "volume", "quote_volume", "trades",
                                   "taker_buy_base", "taker_buy_quote"])
        df.index = pd.to_datetime([int(x[0]) for x in rows], unit="ms", utc=True)
        df.index.name = "open_time"
        return df

    def _init(self, symbol, tf):
        try:
            df = loaders.load_ohlcv("perp", tf, allow_holdout=True, symbol=symbol)
            ev = loaders.load_funding(None, allow_holdout=True, symbol=symbol)
        except FileNotFoundError:
            df, ev = pd.DataFrame(), pd.Series(dtype=float)
        self.hist[(symbol, tf)] = df
        self.fund.setdefault(symbol, ev)

    def refresh(self, symbol, tf) -> tuple[pd.DataFrame, tuple | None]:
        """Returns (closed bars incl. funding column, forming (open_time_s, open) or None)."""
        key = (symbol, tf)
        if key not in self.hist:
            self._init(symbol, tf)
        df = self.hist[key]
        step_ms = tf_seconds(tf) * 1000
        start = int(df.index[-1].timestamp() * 1000) + step_ms if len(df) else self.server_ms - 1500 * step_ms
        new = []
        while True:                                                 # tail, paginated
            rows = self.get("/fapi/v1/klines", symbol=symbol, interval=tf, startTime=start, limit=1500)
            if not rows:
                break
            new += rows
            nxt = int(rows[-1][0]) + step_ms
            if len(rows) < 1500 or nxt > self.server_ms:
                break
            start = nxt
        forming, closed_rows = None, []
        for r in new:
            if int(r[6]) < self.server_ms:                          # closeTime already passed
                closed_rows.append(r)
            else:
                forming = (int(r[0]) // 1000, float(r[1]))
        if closed_rows:
            add = self._rows(closed_rows)
            df = pd.concat([df, add[~add.index.isin(df.index)]]).sort_index()
        df = df[(df.index.asi8 // 10**6) + step_ms <= self.server_ms]   # drop anything not closed
        self.hist[key] = df
        if len(df) > 1:
            gaps = np.diff(df.index.asi8 // 10**6)
            if (gaps != step_ms).any():
                bad = df.index[1:][gaps != step_ms][-1]
                raise RuntimeError(f"gap in {symbol} {tf} history before {bad}")
        ev = self.fund[symbol]                                      # funding events since the last known
        fstart = int(ev.index[-1].timestamp() * 1000) + 1 if len(ev) else int(df.index[0].timestamp() * 1000)
        rows = self.get("/fapi/v1/fundingRate", symbol=symbol, startTime=fstart, limit=1000)
        if rows:
            add = pd.Series([float(x["fundingRate"]) for x in rows],
                            index=pd.to_datetime([int(x["fundingTime"]) for x in rows], unit="ms", utc=True).floor("s"))
            ev = pd.concat([ev, add[~add.index.isin(ev.index)]]).sort_index()
            self.fund[symbol] = ev
        out = df.copy()
        out["funding"] = align_funding(out.index, ev, tf)
        return out, forming


# ------------------------------------------------------------------ service loop
def run_cycle(con, market: Market, done: dict) -> dict:
    t0 = time.time()
    errors, last_bars, new_rows, cfg = [], {}, 0, []
    try:
        cfg = load_config()
        market.sync_time()
    except Exception as e:  # noqa: BLE001
        errors.append(f"setup: {e!r}")
        cfg = []
    cache: dict = {}
    for strat in cfg:
        sid = strat["id"]
        try:
            row = con.execute("SELECT spec FROM strategies WHERE id=?", (sid,)).fetchone()
            if row:                                                 # spec is frozen at registration
                if json.loads(row[0]) != strat["spec"]:
                    errors.append(f"{sid}: config differs from the registered spec (using registered)")
                strat = {**strat, "spec": json.loads(row[0])}
            key = (strat["symbol"], strat["tf"])
            if key not in cache:
                cache[key] = market.refresh(*key)
            bars, forming = cache[key]
            last_bars[sid] = int(bars.index[-1].timestamp()) if len(bars) else None
            sig = (last_bars[sid], forming[0] if forming else None)
            if done.get(sid) == sig:
                continue
            new_rows += process(con, strat, bars, forming)
            done[sid] = sig
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sid}: {e!r}")
            traceback.print_exc()
    con.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?)",
                (int(t0), _now_iso(), time.time() - t0, len(cfg), new_rows, json.dumps(errors),
                 json.dumps(last_bars)))
    con.commit()
    return {"new_rows": new_rows, "errors": errors}


def _lock(path: Path):
    f = open(path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("paper service already running (lock held)")
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    lock = _lock(PAPER_DIR / "paper.lock")  # noqa: F841
    con, market, done = connect(), Market(), {}
    while True:
        t0 = time.time()
        r = run_cycle(con, market, done)
        print(f"{_now_iso()} new_rows={r['new_rows']} errors={r['errors']}", flush=True)
        if a.once:
            return
        time.sleep(max(1.0, CYCLE_SECONDS - (time.time() - t0)))


if __name__ == "__main__":
    main()
