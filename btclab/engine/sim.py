"""Numba single-asset portfolio simulator.

Semantics (see docs/plan.md, section C):

* A signal evaluated on the close of bar ``i`` fills at ``open[i+1]``. Nothing
  ever fills at the same bar's close.
* Stop-loss, take-profit and trailing stop are checked intrabar against
  ``high``/``low`` of every bar the position is open, including the entry bar.
  When both a stop and a target could hit in the same bar, the stop is assumed
  to hit first (worst case). If the bar opens beyond a stop level the stop
  fills at the open (gap-through). A take-profit that gaps favourably also fills
  at the open.
* Stop distances are fixed at entry from the ATR of the *signal* bar (no
  lookahead) or from a percentage of the entry price. If both an ATR multiple
  and a percentage are given, the tighter one wins. The trailing stop ratchets
  from the running extreme of previous bars (never from the current bar's own
  extreme, so it cannot be hit "before" the price that set it).
* Funding: the rate array is aligned to bars and non-zero only on funding bars.
  Funding is settled on the position that is open right after the fills at the
  open of that bar, on notional ``qty * open[i]``. Longs pay when the rate is
  positive, shorts receive; the opposite for negative rates.
* Fees and slippage are charged on every fill, including stop/target fills.
  Fixed slippage is in basis points, optionally plus a per-bar extra fraction
  (``slip_extra``, e.g. ``0.1 * atr / close`` for ATR-scaled slippage).
* Position sizing: notional = equity * size_frac * leverage at the time of
  entry. No pyramiding: an entry signal while in a position is ignored, unless
  it is the opposite side and ``reverse_on_signal`` is set, in which case the
  position is closed and reversed at the next open (two fills). An explicit
  exit signal plus an opposite entry on the same bar always exits and re-enters
  at the next open (that is an explicit instruction, not pyramiding). If both
  entry signals fire on the same bar while flat, both are ignored.
* Time stop ``time_stop = k``: a position entered at ``open[e]`` is closed at
  ``open[e + k]`` unless something else closed it first.
* A position still open on the last bar is marked to the last close and
  recorded as a trade with ``exit_reason = EXIT_END`` and no exit fee, so a
  buy-and-hold run reproduces ``close / entry_open - fee``.
* If equity drops to zero or below the account is bankrupt: equity is clamped
  to zero and no further trades are taken.

Trade record fields: entry_idx, exit_idx, side (+1/-1), entry_px, exit_px,
pnl (net quote PnL incl. fees and funding), pnl_pct (net return on entry
notional, fraction), fees (quote), funding (quote, positive = paid),
exit_reason (EXIT_* code), bars (exit_idx - entry_idx).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

EXIT_SIGNAL = 0
EXIT_STOP = 1
EXIT_TAKE_PROFIT = 2
EXIT_TRAIL = 3
EXIT_TIME = 4
EXIT_REVERSE = 5
EXIT_END = 6

EXIT_REASON_NAMES = {
    EXIT_SIGNAL: "signal",
    EXIT_STOP: "stop_loss",
    EXIT_TAKE_PROFIT: "take_profit",
    EXIT_TRAIL: "trailing_stop",
    EXIT_TIME: "time_stop",
    EXIT_REVERSE: "reverse",
    EXIT_END: "end_of_data",
}

TRADE_DTYPE = np.dtype(
    [
        ("entry_idx", np.int64),
        ("exit_idx", np.int64),
        ("side", np.int8),
        ("entry_px", np.float64),
        ("exit_px", np.float64),
        ("pnl", np.float64),
        ("pnl_pct", np.float64),
        ("fees", np.float64),
        ("funding", np.float64),
        ("exit_reason", np.int8),
        ("bars", np.int64),
    ]
)


@dataclass
class SimParams:
    """Scalar parameters of the simulator. Zero disables an optional stop."""

    fee_rate: float = 0.0005        # per side, fraction of notional
    slippage_bp: float = 2.0        # fixed slippage per side, basis points
    size_frac: float = 1.0          # fraction of equity committed per trade
    leverage: float = 1.0           # notional multiplier
    sl_atr: float = 0.0             # stop-loss distance in ATR multiples
    tp_atr: float = 0.0             # take-profit distance in ATR multiples
    trail_atr: float = 0.0          # trailing-stop distance in ATR multiples
    sl_pct: float = 0.0             # stop-loss as fraction of entry price
    tp_pct: float = 0.0             # take-profit as fraction of entry price
    trail_pct: float = 0.0          # trailing stop as fraction of running extreme
    time_stop: int = 0              # max bars held (0 = off)
    allow_short: bool = True
    reverse_on_signal: bool = False
    init_cash: float = 1.0

    def as_args(self) -> tuple:
        return (
            float(self.fee_rate),
            float(self.slippage_bp),
            float(self.size_frac),
            float(self.leverage),
            float(self.sl_atr),
            float(self.tp_atr),
            float(self.trail_atr),
            float(self.sl_pct),
            float(self.tp_pct),
            float(self.trail_pct),
            int(self.time_stop),
            bool(self.allow_short),
            bool(self.reverse_on_signal),
            float(self.init_cash),
        )


@dataclass
class SimResult:
    equity: np.ndarray            # per bar, mark-to-market at close
    position: np.ndarray          # per bar, -1/0/1 (state at the close)
    trades: np.ndarray            # structured array, TRADE_DTYPE
    params: SimParams = field(default_factory=SimParams)

    @property
    def n_trades(self) -> int:
        return int(self.trades.shape[0])


@njit(cache=True, nogil=True)
def _simulate(
    open_, high, low, close, atr, funding, slip_extra,
    entry_long, entry_short, exit_long, exit_short,
    fee_rate, slippage_bp, size_frac, leverage,
    sl_atr, tp_atr, trail_atr, sl_pct, tp_pct, trail_pct,
    time_stop, allow_short, reverse_on_signal, init_cash,
):
    n = open_.shape[0]
    equity = np.empty(n, dtype=np.float64)
    position = np.zeros(n, dtype=np.int8)

    t_entry_idx = np.empty(n, dtype=np.int64)
    t_exit_idx = np.empty(n, dtype=np.int64)
    t_side = np.empty(n, dtype=np.int8)
    t_entry_px = np.empty(n, dtype=np.float64)
    t_exit_px = np.empty(n, dtype=np.float64)
    t_pnl = np.empty(n, dtype=np.float64)
    t_pnl_pct = np.empty(n, dtype=np.float64)
    t_fees = np.empty(n, dtype=np.float64)
    t_funding = np.empty(n, dtype=np.float64)
    t_reason = np.empty(n, dtype=np.int8)
    nt = 0

    slip_fixed = slippage_bp / 10000.0
    has_atr = atr.shape[0] == n
    has_funding = funding.shape[0] == n
    has_slip_extra = slip_extra.shape[0] == n
    use_sl = (sl_atr > 0.0 and has_atr) or sl_pct > 0.0
    use_tp = (tp_atr > 0.0 and has_atr) or tp_pct > 0.0
    use_trail = (trail_atr > 0.0 and has_atr) or trail_pct > 0.0

    cash = init_cash            # equity when flat; equity at entry while in a trade
    side = 0                    # -1, 0, +1
    qty = 0.0
    entry_px = 0.0
    entry_idx = -1
    entry_fee = 0.0
    trade_fund = 0.0
    sl_level = 0.0
    tp_level = 0.0
    trail_dist = 0.0            # absolute distance (ATR mode) ...
    trail_frac = 0.0            # ... or fraction of the extreme (pct mode)
    trail_level = 0.0
    extreme = 0.0               # running best price since entry (prev bars)

    pending_exit = False
    pending_reason = EXIT_SIGNAL
    pending_entry = 0           # 0 none, +1 long, -1 short
    bankrupt = False

    for i in range(n):
        o = open_[i]
        slip = slip_fixed
        if has_slip_extra:
            slip += slip_extra[i]

        # ---- 1. fills at the open ---------------------------------------
        if side != 0 and pending_exit:
            px = o * (1.0 - slip) if side == 1 else o * (1.0 + slip)
            fee = qty * px * fee_rate
            gross = (px - entry_px) * qty * side
            fees = entry_fee + fee
            net = gross - fees - trade_fund
            t_entry_idx[nt] = entry_idx
            t_exit_idx[nt] = i
            t_side[nt] = side
            t_entry_px[nt] = entry_px
            t_exit_px[nt] = px
            t_pnl[nt] = net
            t_pnl_pct[nt] = net / (qty * entry_px)
            t_fees[nt] = fees
            t_funding[nt] = trade_fund
            t_reason[nt] = pending_reason
            nt += 1
            cash = cash + net
            side = 0
            qty = 0.0
            trade_fund = 0.0
            entry_fee = 0.0
            if cash <= 0.0:
                cash = 0.0
                bankrupt = True
                pending_entry = 0
        pending_exit = False

        if side == 0 and pending_entry != 0 and not bankrupt:
            s = pending_entry
            px = o * (1.0 + slip) if s == 1 else o * (1.0 - slip)
            notional = cash * size_frac * leverage
            qty = notional / px
            entry_fee = notional * fee_rate
            entry_px = px
            entry_idx = i
            side = s
            trade_fund = 0.0
            extreme = px
            # stop levels, fixed at entry from the signal bar's ATR (i-1)
            a = atr[i - 1] if (has_atr and i > 0) else 0.0
            if use_sl:
                d = 1e300
                if sl_atr > 0.0 and a > 0.0:
                    d = sl_atr * a
                if sl_pct > 0.0:
                    d2 = sl_pct * px
                    if d2 < d:
                        d = d2
                sl_level = px - d if s == 1 else px + d
            if use_tp:
                d = 1e300
                if tp_atr > 0.0 and a > 0.0:
                    d = tp_atr * a
                if tp_pct > 0.0:
                    d2 = tp_pct * px
                    if d2 < d:
                        d = d2
                tp_level = px + d if s == 1 else px - d
            if use_trail:
                trail_dist = 1e300
                trail_frac = 0.0
                if trail_atr > 0.0 and a > 0.0:
                    trail_dist = trail_atr * a
                if trail_pct > 0.0:
                    trail_frac = trail_pct
                    if trail_pct * px < trail_dist:
                        trail_dist = 1e300  # pct mode is the tighter one
                if trail_dist < 1e300:
                    trail_level = px - trail_dist if s == 1 else px + trail_dist
                else:
                    trail_level = px * (1.0 - trail_frac) if s == 1 else px * (1.0 + trail_frac)
        pending_entry = 0

        # ---- 2. funding on the open position (funding time = bar open) ---
        if side != 0 and has_funding:
            f = funding[i]
            if f != 0.0:
                trade_fund += f * qty * o * side

        # ---- 3. intrabar stops, worst case: stop before target ----------
        if side != 0 and (use_sl or use_tp or use_trail):
            h = high[i]
            lo = low[i]
            hit = 0
            px = 0.0
            if side == 1:
                stop_lvl = -1e300
                stop_kind = 0
                if use_sl:
                    stop_lvl = sl_level
                    stop_kind = EXIT_STOP
                if use_trail and trail_level > stop_lvl:
                    stop_lvl = trail_level
                    stop_kind = EXIT_TRAIL
                if stop_kind != 0 and lo <= stop_lvl:
                    hit = stop_kind
                    px = o if o <= stop_lvl else stop_lvl
                    px = px * (1.0 - slip)
                elif use_tp and h >= tp_level:
                    hit = EXIT_TAKE_PROFIT
                    px = o if o >= tp_level else tp_level
                    px = px * (1.0 - slip)
            else:
                stop_lvl = 1e300
                stop_kind = 0
                if use_sl:
                    stop_lvl = sl_level
                    stop_kind = EXIT_STOP
                if use_trail and trail_level < stop_lvl:
                    stop_lvl = trail_level
                    stop_kind = EXIT_TRAIL
                if stop_kind != 0 and h >= stop_lvl:
                    hit = stop_kind
                    px = o if o >= stop_lvl else stop_lvl
                    px = px * (1.0 + slip)
                elif use_tp and lo <= tp_level:
                    hit = EXIT_TAKE_PROFIT
                    px = o if o <= tp_level else tp_level
                    px = px * (1.0 + slip)
            if hit != 0:
                fee = qty * px * fee_rate
                gross = (px - entry_px) * qty * side
                fees = entry_fee + fee
                net = gross - fees - trade_fund
                t_entry_idx[nt] = entry_idx
                t_exit_idx[nt] = i
                t_side[nt] = side
                t_entry_px[nt] = entry_px
                t_exit_px[nt] = px
                t_pnl[nt] = net
                t_pnl_pct[nt] = net / (qty * entry_px)
                t_fees[nt] = fees
                t_funding[nt] = trade_fund
                t_reason[nt] = hit
                nt += 1
                cash = cash + net
                side = 0
                qty = 0.0
                trade_fund = 0.0
                entry_fee = 0.0
                if cash <= 0.0:
                    cash = 0.0
                    bankrupt = True
            elif use_trail:
                # ratchet from this bar's extreme for the *next* bars
                if side == 1:
                    if h > extreme:
                        extreme = h
                        lvl = extreme - trail_dist if trail_dist < 1e300 else extreme * (1.0 - trail_frac)
                        if lvl > trail_level:
                            trail_level = lvl
                else:
                    if lo < extreme:
                        extreme = lo
                        lvl = extreme + trail_dist if trail_dist < 1e300 else extreme * (1.0 + trail_frac)
                        if lvl < trail_level:
                            trail_level = lvl

        # ---- 4. mark to market at the close ------------------------------
        c = close[i]
        if side != 0:
            equity[i] = cash + (c - entry_px) * qty * side - entry_fee - trade_fund
            if equity[i] <= 0.0:
                equity[i] = 0.0
        else:
            equity[i] = cash
        position[i] = side

        # ---- 5. signals on the close -> orders for the next open --------
        if bankrupt:
            continue
        el = entry_long[i]
        es = entry_short[i] and allow_short
        if side == 1:
            if exit_long[i]:
                pending_exit = True
                pending_reason = EXIT_SIGNAL
                if es:
                    pending_entry = -1
            elif es and reverse_on_signal:
                pending_exit = True
                pending_reason = EXIT_REVERSE
                pending_entry = -1
        elif side == -1:
            if exit_short[i]:
                pending_exit = True
                pending_reason = EXIT_SIGNAL
                if el:
                    pending_entry = 1
            elif el and reverse_on_signal:
                pending_exit = True
                pending_reason = EXIT_REVERSE
                pending_entry = 1
        else:
            if el and not es:
                pending_entry = 1
            elif es and not el:
                pending_entry = -1
        if side != 0 and time_stop > 0 and not pending_exit:
            if i - entry_idx + 1 >= time_stop:
                pending_exit = True
                pending_reason = EXIT_TIME

    # ---- open position at the end: mark at the last close, no exit fee ---
    if side != 0:
        px = close[n - 1]
        gross = (px - entry_px) * qty * side
        net = gross - entry_fee - trade_fund
        t_entry_idx[nt] = entry_idx
        t_exit_idx[nt] = n - 1
        t_side[nt] = side
        t_entry_px[nt] = entry_px
        t_exit_px[nt] = px
        t_pnl[nt] = net
        t_pnl_pct[nt] = net / (qty * entry_px)
        t_fees[nt] = entry_fee
        t_funding[nt] = trade_fund
        t_reason[nt] = EXIT_END
        nt += 1

    return (
        equity, position,
        t_entry_idx[:nt], t_exit_idx[:nt], t_side[:nt], t_entry_px[:nt],
        t_exit_px[:nt], t_pnl[:nt], t_pnl_pct[:nt], t_fees[:nt],
        t_funding[:nt], t_reason[:nt],
    )


_EMPTY_F64 = np.empty(0, dtype=np.float64)


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _bool(a, n: int) -> np.ndarray:
    if a is None:
        return np.zeros(n, dtype=np.bool_)
    return np.ascontiguousarray(a, dtype=np.bool_)


def simulate(
    open_, high, low, close,
    entry_long=None, entry_short=None, exit_long=None, exit_short=None,
    *,
    atr=None, funding=None, slip_extra=None,
    params: SimParams | None = None,
    **overrides,
) -> SimResult:
    """Run the simulator on numpy arrays. See the module docstring for semantics.

    ``params`` is a :class:`SimParams`; any field can also be passed as a
    keyword override (``simulate(..., fee_rate=0.001, sl_atr=2.0)``). ``atr``
    is needed for ATR-based stops and ATR-scaled slippage; ``funding`` is a
    per-bar rate array (zero on non-funding bars); ``slip_extra`` is a per-bar
    extra slippage fraction added to ``slippage_bp``.
    """
    p = params if params is not None else SimParams()
    if overrides:
        p = SimParams(**{**p.__dict__, **overrides})

    o = _f64(open_)
    n = o.shape[0]
    h = _f64(high)
    lo = _f64(low)
    c = _f64(close)
    if not (h.shape[0] == lo.shape[0] == c.shape[0] == n):
        raise ValueError("open/high/low/close must have the same length")
    atr_a = _f64(atr) if atr is not None else _EMPTY_F64
    fund_a = _f64(funding) if funding is not None else _EMPTY_F64
    slip_a = _f64(slip_extra) if slip_extra is not None else _EMPTY_F64
    for name, arr in (("atr", atr_a), ("funding", fund_a), ("slip_extra", slip_a)):
        if arr.shape[0] not in (0, n):
            raise ValueError(f"{name} must have length {n} or be None")

    out = _simulate(
        o, h, lo, c, atr_a, fund_a, slip_a,
        _bool(entry_long, n), _bool(entry_short, n), _bool(exit_long, n), _bool(exit_short, n),
        *p.as_args(),
    )
    equity, position = out[0], out[1]
    nt = out[2].shape[0]
    trades = np.empty(nt, dtype=TRADE_DTYPE)
    trades["entry_idx"] = out[2]
    trades["exit_idx"] = out[3]
    trades["side"] = out[4]
    trades["entry_px"] = out[5]
    trades["exit_px"] = out[6]
    trades["pnl"] = out[7]
    trades["pnl_pct"] = out[8]
    trades["fees"] = out[9]
    trades["funding"] = out[10]
    trades["exit_reason"] = out[11]
    trades["bars"] = out[3] - out[2]
    return SimResult(equity=equity, position=position, trades=trades, params=p)


def warmup() -> None:
    """Trigger JIT compilation (or load the cached kernel) on a tiny input."""
    z = np.ones(4)
    simulate(z, z, z, z)
