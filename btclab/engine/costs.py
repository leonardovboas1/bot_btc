"""Cost presets for the simulator.

A :class:`CostModel` bundles the per-side fee, the fixed slippage and whether
shorting is allowed. ``CostModel.sim_kwargs()`` returns the matching keyword
arguments for :func:`btclab.engine.sim.simulate`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostModel:
    name: str
    fee_rate: float          # per side, fraction of notional (0.0005 = 0.05%)
    slippage_bp: float = 2.0  # fixed slippage per side, basis points (plan C: 2 bp)
    allow_short: bool = True
    verified: bool = True     # False when the fee has not been checked against the venue
    note: str = ""

    def sim_kwargs(self) -> dict:
        """Keyword arguments for ``simulate(...)``."""
        return {
            "fee_rate": self.fee_rate,
            "slippage_bp": self.slippage_bp,
            "allow_short": self.allow_short,
        }

    def to_dict(self) -> dict:
        return asdict(self)


# Binance USDT-M perpetual, regular user, taker: 0.05% per side.
BINANCE_PERP_TAKER = CostModel(
    name="binance_perp_taker",
    fee_rate=0.0005,
    note="Binance USDT-M perp, VIP0 taker 0.05%/side.",
)

# The user's Pine strategy() uses commission_value=0.1 (%), so 0.1% per side.
USER_TV_DEFAULT = CostModel(
    name="user_tv_default",
    fee_rate=0.001,
    note="Matches the user's TradingView strategy (commission 0.1%/side).",
)

# Spot: 0.1% per side, long only (no funding, no shorts).
SPOT = CostModel(
    name="spot",
    fee_rate=0.001,
    allow_short=False,
    note="Binance spot VIP0 taker 0.1%/side, long only.",
)

# Hyperliquid perps, Tier 0 base taker 0.045% (maker 0.015%).
# Verified 2026-11-10 at https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees
HYPERLIQUID_TAKER = CostModel(
    name="hyperliquid_taker",
    fee_rate=0.00045,
    verified=True,
    note="Hyperliquid perps Tier 0 taker 0.045%/side (docs, 2026-11-10).",
)

PRESETS: dict[str, CostModel] = {
    m.name: m for m in (BINANCE_PERP_TAKER, USER_TV_DEFAULT, SPOT, HYPERLIQUID_TAKER)
}
