"""
src/backtest/strategy.py

Mechanical trading rules: the rule-based baseline strategy.

Strategy logic (V1)
───────────────────
On any day where a ticker appears on the scanner watchlist:

  FADE (short the overreaction):
    - Trigger: catalyst + herd_score >= threshold + gap_pct >= gap_threshold
    - Entry:   at the open (or first bar after a defined entry window)
    - Direction: SHORT (fading the gap in high-herd-score names)
    - Stop:   entry × (1 + stop_pct)  (e.g. 2%)
    - Target: entry × (1 - target_pct) (e.g. 4%)
    - Exit:   target, stop, or EOD  — whichever comes first

  FOLLOW (ride the momentum):
    - Trigger: catalyst + herd_score >= threshold + volume spike early
    - Entry:   first 30-min breakout above pre-market high
    - Direction: LONG
    - Stop:   pre-market low
    - Target: entry × (1 + target_pct)
    - Exit:   target, stop, or EOD

Signal selection: if herd_stage == 'saturated' → FADE; else → FOLLOW.

This is intentionally simple (no ML, no complex rules) to give a
clean baseline for comparing against the Phase 2 RL agent.

Signals are generated per-bar during the engine event loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.backtest.execution import Direction, Order, PositionType
from src.backtest.leverage import (
    PositionSpec,
    size_debit_spread,
    size_equity,
    size_margin,
)
from src.backtest.risk_manager import RiskManager, RiskState
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

_BT     = settings["backtest"]
_SCAN   = settings["scanner"]

SIGNAL_FADE   = "fade"
SIGNAL_FOLLOW = "follow"
SIGNAL_NONE   = "none"


@dataclass
class WatchlistEntry:
    """A single row from the scanner_watchlist / pre-market scan."""
    ticker: str
    date: str
    catalyst_type: str
    gap_pct: float
    relative_volume: float
    herd_score: float
    herd_direction: str
    herd_stage: str
    market_cap: float
    pre_market_high: float | None = None
    pre_market_low: float | None = None
    open_price: float | None = None


@dataclass
class Signal:
    ticker: str
    date: str
    signal_type: str              # FADE / FOLLOW / NONE
    direction: Direction
    position_type: PositionType
    entry_trigger: str            # "open" | "breakout"
    breakout_level: float | None  # relevant for FOLLOW breakout entry
    stop_price: float | None
    target_price: float | None
    herd_score: float
    catalyst_type: str


class BaseStrategy:
    """
    Abstract base — subclass to override signal_for_entry and/or
    signal_for_exit for custom rules.
    """

    def signal_for_entry(
        self,
        entry: WatchlistEntry,
        current_price: float,
        current_time: str,
    ) -> Signal:
        raise NotImplementedError

    def size_order(
        self,
        signal: Signal,
        state: RiskState,
        risk_manager: RiskManager,
        current_price: float,
    ) -> Order | None:
        raise NotImplementedError


class MechanicalStrategy(BaseStrategy):
    """
    Rule-based baseline strategy.

    Configuration (all in settings.yaml under backtest):
      fade_herd_score_min  (default 6.0)
      follow_herd_score_min (default 4.0)
      gap_min_pct          (default 0.02 = 2%)
      entry_window_minutes (default 30  — breakout entry only)
      position_type        "equity" | "margin" | "debit_spread"  (default "equity")
    """

    def __init__(self) -> None:
        self._fade_min_score   = _BT.get("fade_herd_score_min", 6.0)
        self._follow_min_score = _BT.get("follow_herd_score_min", 4.0)
        self._gap_min          = _SCAN.get("min_gap_pct", 0.02)
        self._entry_window     = _BT.get("entry_window_minutes", 30)
        self._pos_type_cfg     = _BT.get("position_type", "equity")
        self._stop_pct         = _BT.get("stop_loss_pct", 0.02)
        self._target_pct       = _BT.get("target_pct", 0.04)

    def _select_signal_type(self, entry: WatchlistEntry) -> str:
        if entry.herd_score < self._follow_min_score:
            return SIGNAL_NONE
        if abs(entry.gap_pct) < self._gap_min:
            return SIGNAL_NONE
        if entry.herd_stage == "saturated" and entry.herd_score >= self._fade_min_score:
            return SIGNAL_FADE
        return SIGNAL_FOLLOW

    def _select_position_type(self, signal_type: str) -> PositionType:
        mapping = {
            "equity": PositionType.EQUITY,
            "margin": PositionType.MARGIN,
            "debit_spread": PositionType.DEBIT_SPREAD,
        }
        return mapping.get(self._pos_type_cfg, PositionType.EQUITY)

    def signal_for_entry(
        self,
        entry: WatchlistEntry,
        current_price: float,
        current_time: str,
    ) -> Signal:
        sig_type = self._select_signal_type(entry)
        if sig_type == SIGNAL_NONE:
            return Signal(
                ticker=entry.ticker, date=entry.date,
                signal_type=SIGNAL_NONE,
                direction=Direction.LONG,
                position_type=PositionType.EQUITY,
                entry_trigger="none",
                breakout_level=None,
                stop_price=None, target_price=None,
                herd_score=entry.herd_score,
                catalyst_type=entry.catalyst_type,
            )

        if sig_type == SIGNAL_FADE:
            direction = Direction.SHORT
            stop   = round(current_price * (1.0 + self._stop_pct), 4)
            target = round(current_price * (1.0 - self._target_pct), 4)
            trigger = "open"
            breakout_level = None
        else:  # FOLLOW
            direction = Direction.LONG
            # Enter on breakout above pre-market high (or open if no PM data)
            breakout_level = entry.pre_market_high or current_price
            # Stop = pre-market low (or entry - stop_pct fallback)
            stop = entry.pre_market_low or round(current_price * (1.0 - self._stop_pct), 4)
            target = round(current_price * (1.0 + self._target_pct), 4)
            trigger = "breakout"

        return Signal(
            ticker=entry.ticker,
            date=entry.date,
            signal_type=sig_type,
            direction=direction,
            position_type=self._select_position_type(sig_type),
            entry_trigger=trigger,
            breakout_level=breakout_level,
            stop_price=stop,
            target_price=target,
            herd_score=entry.herd_score,
            catalyst_type=entry.catalyst_type,
        )

    def size_order(
        self,
        signal: Signal,
        state: RiskState,
        risk_manager: RiskManager,
        current_price: float,
    ) -> Order | None:
        if signal.signal_type == SIGNAL_NONE:
            return None

        ok, reason = risk_manager.can_enter(state)
        if not ok:
            log.debug("entry_blocked", ticker=signal.ticker, reason=reason)
            return None

        max_dollar = risk_manager.max_dollar_exposure(state)
        pos_size_pct = _BT.get("position_size_pct", 0.20)

        if signal.position_type == PositionType.MARGIN:
            spec = size_margin(state.current_capital, current_price, pos_size_pct)
        elif signal.position_type == PositionType.DEBIT_SPREAD:
            spec = size_debit_spread(state.current_capital, current_price, signal.direction)
        else:
            spec = size_equity(state.current_capital, current_price, pos_size_pct)

        if spec.shares_or_contracts <= 0:
            log.debug("zero_size", ticker=signal.ticker, capital=state.current_capital)
            return None

        # Determine time for the order
        order_time = "09:30:00" if signal.entry_trigger == "open" else "10:00:00"

        return Order(
            ticker=signal.ticker,
            date=signal.date,
            time=order_time,
            direction=signal.direction,
            shares=spec.shares_or_contracts,
            position_type=signal.position_type,
            leverage_multiple=spec.leverage_multiple,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            strategy_signal=signal.signal_type,
            herd_score_at_entry=signal.herd_score,
            catalyst_type=signal.catalyst_type,
        )
