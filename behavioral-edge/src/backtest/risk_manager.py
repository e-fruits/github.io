"""
src/backtest/risk_manager.py

Position sizing, daily loss limits, and circuit breakers.

Rules enforced
──────────────
  1. Max position size as % of current capital
     (settings.backtest.position_size_pct, default 20%)

  2. Max concurrent open positions
     (settings.backtest.max_positions, default 5)

  3. Daily loss limit as % of starting-day capital
     (settings.backtest.max_daily_loss_pct, default 2%)
     → triggers circuit breaker: no new entries for the rest of the day

  4. Per-trade stop-loss enforcement (handled by the engine, but
     RiskManager provides the stop price calculation)

  5. EOD forced exit at settings.backtest.eod_close_time (default 15:55)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backtest.execution import Direction, Fill
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

_BT = settings["backtest"]


@dataclass
class RiskState:
    """Mutable state carried through a single trading day."""
    date: str
    starting_capital: float
    current_capital: float
    open_positions: list[Fill] = field(default_factory=list)
    closed_today: list[Fill] = field(default_factory=list)
    circuit_breaker_triggered: bool = False
    daily_pnl: float = 0.0

    @property
    def num_open(self) -> int:
        return len(self.open_positions)

    @property
    def daily_return_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return self.daily_pnl / self.starting_capital

    @property
    def max_drawdown_intraday(self) -> float:
        """Maximum unrealised loss from starting capital, intraday."""
        return min(self.daily_pnl, 0.0)


class RiskManager:
    def __init__(self) -> None:
        self._max_positions    = _BT.get("max_positions", 5)
        self._pos_size_pct     = _BT.get("position_size_pct", 0.20)
        self._max_daily_loss   = _BT.get("max_daily_loss_pct", 0.02)
        self._eod_close_time   = _BT.get("eod_close_time", "15:55")
        self._stop_loss_pct    = _BT.get("stop_loss_pct", 0.02)    # default 2% stop
        self._target_pct       = _BT.get("target_pct", 0.04)       # default 4% target (2:1 R/R)

    # ── Sizing ──────────────────────────────────────────────────────────────

    def max_dollar_exposure(self, state: RiskState) -> float:
        """Maximum notional value for a new position."""
        return state.current_capital * self._pos_size_pct

    def size_shares(
        self,
        state: RiskState,
        entry_price: float,
        leverage: float = 1.0,
    ) -> int:
        """
        Return the number of shares to trade given current state and
        the entry price.  Applies leverage so that the *effective*
        notional (shares × price) does not exceed max_dollar_exposure.
        """
        if entry_price <= 0:
            return 0
        max_notional = self.max_dollar_exposure(state)
        # With leverage, we need less capital per notional dollar
        capital_per_notional = 1.0 / leverage
        affordable_notional = max_notional / capital_per_notional
        return max(int(affordable_notional / entry_price), 0)

    # ── Entry checks ────────────────────────────────────────────────────────

    def can_enter(self, state: RiskState) -> tuple[bool, str]:
        """
        Return (True, "") if a new position is allowed,
        or (False, reason) if blocked.
        """
        if state.circuit_breaker_triggered:
            return False, "circuit_breaker"
        if state.num_open >= self._max_positions:
            return False, f"max_positions ({self._max_positions})"
        return True, ""

    def check_circuit_breaker(self, state: RiskState) -> bool:
        """
        Trigger circuit breaker if daily loss exceeds the limit.
        Mutates state.circuit_breaker_triggered.
        Returns True if triggered.
        """
        loss_pct = -state.daily_return_pct  # positive = loss
        if loss_pct >= self._max_daily_loss:
            if not state.circuit_breaker_triggered:
                state.circuit_breaker_triggered = True
                log.warning(
                    "circuit_breaker_triggered",
                    date=state.date,
                    daily_loss_pct=round(loss_pct * 100, 2),
                )
            return True
        return False

    # ── Stop / target prices ────────────────────────────────────────────────

    def calculate_stop(self, entry_price: float, direction: Direction) -> float:
        """Default stop: 2% against position."""
        if direction == Direction.LONG:
            return round(entry_price * (1.0 - self._stop_loss_pct), 4)
        return round(entry_price * (1.0 + self._stop_loss_pct), 4)

    def calculate_target(self, entry_price: float, direction: Direction) -> float:
        """Default target: 4% in favour of position (2:1 R/R)."""
        if direction == Direction.LONG:
            return round(entry_price * (1.0 + self._target_pct), 4)
        return round(entry_price * (1.0 - self._target_pct), 4)

    # ── EOD check ───────────────────────────────────────────────────────────

    def is_eod(self, current_time: str) -> bool:
        """Return True if *current_time* (HH:MM) is at or past EOD close time."""
        return current_time >= self._eod_close_time

    # ── Position management ─────────────────────────────────────────────────

    def register_open(self, state: RiskState, fill: Fill) -> None:
        state.open_positions.append(fill)
        # Deduct capital (margin/spread accounting is simplified: full notional reserved)
        state.current_capital -= fill.entry_price * fill.shares / fill.leverage_multiple

    def register_close(self, state: RiskState, fill: Fill) -> None:
        state.open_positions = [p for p in state.open_positions if p is not fill]
        state.closed_today.append(fill)
        # Return capital + PnL
        state.current_capital += fill.entry_price * fill.shares / fill.leverage_multiple
        state.current_capital += fill.pnl_dollars
        state.daily_pnl       += fill.pnl_dollars
        self.check_circuit_breaker(state)

    # ── Intraday stop/target checks ─────────────────────────────────────────

    def check_exits(
        self,
        state: RiskState,
        current_time: str,
        price_by_ticker: dict[str, float],
    ) -> list[tuple[Fill, float, str]]:
        """
        Scan open positions for stop/target/EOD triggers.
        Returns list of (fill, exit_price, exit_reason) for positions
        that should be closed at *current_time*.
        """
        exits: list[tuple[Fill, float, str]] = []
        is_eod = self.is_eod(current_time)

        for fill in list(state.open_positions):
            price = price_by_ticker.get(fill.ticker)
            if price is None:
                continue

            if is_eod:
                exits.append((fill, price, "time_stop"))
                continue

            if fill.direction == Direction.LONG:
                if fill.stop_price and price <= fill.stop_price:
                    exits.append((fill, fill.stop_price, "stop"))
                elif fill.target_price and price >= fill.target_price:
                    exits.append((fill, fill.target_price, "target"))
            else:  # SHORT
                if fill.stop_price and price >= fill.stop_price:
                    exits.append((fill, fill.stop_price, "stop"))
                elif fill.target_price and price <= fill.target_price:
                    exits.append((fill, fill.target_price, "target"))

        return exits
