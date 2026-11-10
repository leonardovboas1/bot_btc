"""Backtest engine: numba portfolio simulator, cost presets and metrics."""

from btclab.engine.costs import (  # noqa: F401
    BINANCE_PERP_TAKER,
    HYPERLIQUID_TAKER,
    SPOT,
    USER_TV_DEFAULT,
    CostModel,
)
from btclab.engine.metrics import compute_metrics, daily_returns  # noqa: F401
from btclab.engine.sim import (  # noqa: F401
    EXIT_END,
    EXIT_REVERSE,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TAKE_PROFIT,
    EXIT_TIME,
    EXIT_TRAIL,
    TRADE_DTYPE,
    SimParams,
    SimResult,
    simulate,
)
