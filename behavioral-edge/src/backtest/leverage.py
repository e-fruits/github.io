"""
src/backtest/leverage.py

Simulates three position types:

  equity        — plain long/short with no leverage.
                  Cost = shares × price.

  margin        — 2× leverage.  Reg-T margin requires 50% of notional
                  as capital.  We charge a simulated overnight interest
                  rate only when positions are held past a configurable
                  cutoff (this system targets intraday, so typically $0).

  debit_spread  — a simplified options debit spread model.
                  Entry cost = net debit × 100 × contracts.
                  Max loss = debit paid.
                  Max gain = spread width − debit.
                  No greeks simulation (V1); just fixed entry/exit model
                  based on moneyness at expiration or price target.

All three types ultimately return a notional_exposure, capital_required,
and max_loss so the risk manager can size correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.backtest.execution import Direction, PositionType
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

_BT = settings["backtest"]
_MARGIN_MULTIPLIER = _BT.get("margin_multiplier", 2.0)
# Annualised margin interest rate (broker charges ~ 8-12% / year)
_MARGIN_RATE_ANNUAL = 0.10
_MARGIN_RATE_DAILY  = _MARGIN_RATE_ANNUAL / 252


@dataclass
class PositionSpec:
    """
    Computed position parameters returned by size_position().
    Fed into execution.simulate_fill() and risk_manager.register_open().
    """
    position_type: PositionType
    shares_or_contracts: int
    capital_required: float         # cash set aside
    notional_exposure: float        # total market exposure
    leverage_multiple: float
    max_loss: float                 # worst-case loss before commissions
    # For spreads:
    spread_width: float = 0.0
    net_debit: float    = 0.0


def size_equity(
    available_capital: float,
    price: float,
    max_position_pct: float = 0.20,
) -> PositionSpec:
    """Plain equity (no leverage)."""
    capital_to_use = available_capital * max_position_pct
    shares = max(int(capital_to_use / price), 0)
    notional = shares * price
    return PositionSpec(
        position_type=PositionType.EQUITY,
        shares_or_contracts=shares,
        capital_required=notional,
        notional_exposure=notional,
        leverage_multiple=1.0,
        max_loss=notional,   # can go to zero in theory
    )


def size_margin(
    available_capital: float,
    price: float,
    max_position_pct: float = 0.20,
    multiplier: float = _MARGIN_MULTIPLIER,
) -> PositionSpec:
    """
    Margin (2× default).  Capital required = notional / multiplier.
    We double the share count vs. equity at the same capital allocation.
    """
    capital_to_use = available_capital * max_position_pct
    notional = capital_to_use * multiplier
    shares = max(int(notional / price), 0)
    actual_notional = shares * price
    capital_required = actual_notional / multiplier
    return PositionSpec(
        position_type=PositionType.MARGIN,
        shares_or_contracts=shares,
        capital_required=capital_required,
        notional_exposure=actual_notional,
        leverage_multiple=multiplier,
        max_loss=actual_notional,   # max loss is full notional (price → 0)
    )


def size_debit_spread(
    available_capital: float,
    atm_price: float,
    direction: Direction,
    spread_width_pct: float = 0.05,   # spread strikes as % of stock price
    debit_pct: float = 0.40,           # debit as % of spread width (typical ATM)
    max_position_pct: float = 0.20,
) -> PositionSpec:
    """
    Simplified debit spread sizing.

    For a LONG (bullish) call debit spread:
      Buy lower strike call, sell higher strike call.
      Net debit ≈ debit_pct × spread_width.
      Max gain  = spread_width − debit (per share) × 100 × contracts.
      Max loss  = debit paid.

    For a SHORT (bearish) put debit spread:
      Buy higher strike put, sell lower strike put.  Same math.

    We model this as: cost = net_debit × 100 × contracts.
    """
    spread_width = round(atm_price * spread_width_pct, 2)
    net_debit    = round(spread_width * debit_pct, 2)

    capital_to_use = available_capital * max_position_pct
    contracts = max(int(capital_to_use / (net_debit * 100)), 0) if net_debit > 0 else 0
    capital_required = contracts * net_debit * 100
    max_gain = contracts * (spread_width - net_debit) * 100

    return PositionSpec(
        position_type=PositionType.DEBIT_SPREAD,
        shares_or_contracts=contracts,
        capital_required=capital_required,
        notional_exposure=capital_required,   # max loss = capital required
        leverage_multiple=round(atm_price / net_debit, 2) if net_debit > 0 else 1.0,
        max_loss=capital_required,
        spread_width=spread_width,
        net_debit=net_debit,
    )


def compute_margin_interest(notional: float, days_held: float = 1.0) -> float:
    """Daily margin interest for *notional* held for *days_held* trading days."""
    borrowed = notional * (1.0 - 1.0 / _MARGIN_MULTIPLIER)
    return borrowed * _MARGIN_RATE_DAILY * days_held


def compute_spread_pnl(
    spec: PositionSpec,
    entry_stock_price: float,
    exit_stock_price: float,
    direction: Direction,
) -> float:
    """
    Simplified spread P&L: linear interpolation between max loss and max gain
    based on how far the stock moved relative to the spread width.

    This is a rough model — a proper model would use BSM greeks.
    Replace with a real options pricing model for live use.
    """
    move = exit_stock_price - entry_stock_price
    if direction == Direction.SHORT:
        move = -move

    max_gain = spec.contracts_max_gain if hasattr(spec, "contracts_max_gain") else (
        spec.shares_or_contracts * (spec.spread_width - spec.net_debit) * 100
    )
    max_loss = -spec.max_loss

    # Fraction of spread width captured
    fraction = min(max(move / spec.spread_width, -1.0), 1.0) if spec.spread_width > 0 else 0.0
    if fraction >= 0:
        return fraction * max_gain
    else:
        return fraction * abs(max_loss)
