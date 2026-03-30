"""
src/backtest/execution.py

Order simulation layer: converts intended trades into fills with
realistic slippage and commission applied.

Slippage models
───────────────
  fixed_pct     — entry price is moved by a fixed percentage against
                  the trade direction (default 0.1%).  Simple and
                  conservative for small-cap names.

  volume_impact — slippage scales with the order's share of daily
                  volume, following a square-root market-impact model:
                    slippage = k × sqrt(order_size / avg_daily_volume)
                  Appropriate for larger size or higher-frequency
                  simulation.  k is calibrated to 0.1 by default.

All fills are returned as Fill dataclasses for downstream consumption
by the engine and risk manager.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

_BT = settings["backtest"]
_SLIPPAGE_MODEL  = _BT.get("slippage_model", "fixed_pct")
_SLIPPAGE_PCT    = _BT.get("slippage_pct", 0.001)
_COMMISSION      = _BT.get("commission_per_share", 0.005)
_VOLUME_IMPACT_K = 0.1  # calibration constant for volume-impact model


class Direction(str, Enum):
    LONG  = "long"
    SHORT = "short"


class PositionType(str, Enum):
    EQUITY       = "equity"
    MARGIN       = "margin"
    DEBIT_SPREAD = "debit_spread"


class ExitReason(str, Enum):
    TARGET          = "target"
    STOP            = "stop"
    TIME_STOP       = "time_stop"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class Order:
    ticker: str
    date: str
    time: str                          # HH:MM:SS
    direction: Direction
    shares: int
    limit_price: float | None = None   # None = market order
    position_type: PositionType = PositionType.EQUITY
    leverage_multiple: float = 1.0
    stop_price: float | None = None
    target_price: float | None = None
    strategy_signal: str = ""
    herd_score_at_entry: float | None = None
    catalyst_type: str = ""


@dataclass
class Fill:
    ticker: str
    date: str
    entry_time: str
    direction: Direction
    entry_price: float
    shares: int
    commission: float
    slippage: float
    stop_price: float | None
    target_price: float | None
    position_type: PositionType
    leverage_multiple: float
    strategy_signal: str
    herd_score_at_entry: float | None
    catalyst_type: str
    # Filled in at close
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: ExitReason = ExitReason.TIME_STOP
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0


def _fixed_pct_slippage(price: float, direction: Direction) -> float:
    """Move price against the trade direction by a fixed percentage."""
    if direction == Direction.LONG:
        return price * (1.0 + _SLIPPAGE_PCT)
    return price * (1.0 - _SLIPPAGE_PCT)


def _volume_impact_slippage(
    price: float,
    direction: Direction,
    shares: int,
    avg_daily_volume: float,
) -> float:
    """Square-root market-impact slippage model."""
    if avg_daily_volume <= 0:
        return _fixed_pct_slippage(price, direction)
    impact = _VOLUME_IMPACT_K * math.sqrt(shares / avg_daily_volume)
    if direction == Direction.LONG:
        return price * (1.0 + impact)
    return price * (1.0 - impact)


def simulate_fill(
    order: Order,
    market_price: float,
    avg_daily_volume: float = 0.0,
) -> Fill:
    """
    Simulate order execution.  *market_price* is the price at the moment
    of intended entry (e.g. the bar's open).

    Returns a Fill with the estimated entry price after slippage and the
    commission amount.
    """
    if _SLIPPAGE_MODEL == "volume_impact" and avg_daily_volume > 0:
        fill_price = _volume_impact_slippage(
            market_price, order.direction, order.shares, avg_daily_volume
        )
    else:
        fill_price = _fixed_pct_slippage(market_price, order.direction)

    commission = order.shares * _COMMISSION

    log.debug(
        "order_filled",
        ticker=order.ticker,
        direction=order.direction.value,
        market_price=market_price,
        fill_price=round(fill_price, 4),
        shares=order.shares,
        commission=round(commission, 2),
    )

    return Fill(
        ticker=order.ticker,
        date=order.date,
        entry_time=order.time,
        direction=order.direction,
        entry_price=round(fill_price, 4),
        shares=order.shares,
        commission=commission,
        slippage=abs(fill_price - market_price) * order.shares,
        stop_price=order.stop_price,
        target_price=order.target_price,
        position_type=order.position_type,
        leverage_multiple=order.leverage_multiple,
        strategy_signal=order.strategy_signal,
        herd_score_at_entry=order.herd_score_at_entry,
        catalyst_type=order.catalyst_type,
    )


def close_fill(
    fill: Fill,
    exit_price: float,
    exit_time: str,
    exit_reason: ExitReason,
) -> Fill:
    """
    Mutate *fill* in place with exit details and compute PnL.
    Returns the same Fill object for chaining.

    PnL is gross (before commissions on the exit leg).
    Net PnL subtracts both entry and exit commissions.
    """
    exit_commission = fill.shares * _COMMISSION

    if fill.direction == Direction.LONG:
        gross_pnl = (exit_price - fill.entry_price) * fill.shares
    else:
        gross_pnl = (fill.entry_price - exit_price) * fill.shares

    net_pnl = gross_pnl - fill.commission - exit_commission

    fill.exit_time    = exit_time
    fill.exit_price   = round(exit_price, 4)
    fill.exit_reason  = exit_reason
    fill.pnl_dollars  = round(net_pnl, 2)
    fill.pnl_pct      = round(net_pnl / (fill.entry_price * fill.shares), 6) if fill.entry_price else 0.0

    return fill
