"""
tests/test_risk_manager.py

Focused tests for the risk manager's position management and
circuit breaker logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.backtest.execution import Direction, ExitReason, Fill, Order, PositionType, simulate_fill, close_fill
from src.backtest.risk_manager import RiskManager, RiskState


def _make_fill(ticker: str, direction=Direction.LONG, entry=50.0, shares=100) -> Fill:
    order = Order(
        ticker=ticker,
        date="2024-01-10",
        time="09:30:00",
        direction=direction,
        shares=shares,
        position_type=PositionType.EQUITY,
        leverage_multiple=1.0,
        stop_price=entry * 0.98 if direction == Direction.LONG else entry * 1.02,
        target_price=entry * 1.04 if direction == Direction.LONG else entry * 0.96,
    )
    return simulate_fill(order, entry)


class TestPositionLifecycle:
    def test_register_open_reduces_capital(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        fill = _make_fill("A", entry=50.0, shares=100)
        rm.register_open(state, fill)
        assert state.current_capital < 100_000.0
        assert state.num_open == 1

    def test_register_close_returns_capital_and_pnl(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        fill = _make_fill("A", entry=50.0, shares=100)
        rm.register_open(state, fill)
        capital_after_open = state.current_capital

        close_fill(fill, 55.0, "11:00:00", ExitReason.TARGET)
        rm.register_close(state, fill)

        assert state.num_open == 0
        assert state.current_capital > capital_after_open
        assert state.daily_pnl == fill.pnl_dollars

    def test_multiple_positions_up_to_max(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 200_000.0, 200_000.0)
        for i in range(5):
            fill = _make_fill(f"T{i}", entry=50.0, shares=10)
            rm.register_open(state, fill)
        assert state.num_open == 5
        ok, _ = rm.can_enter(state)
        assert ok is False


class TestExitChecks:
    def test_stop_triggered_long(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        fill = _make_fill("A", Direction.LONG, entry=50.0, shares=100)
        fill.stop_price = 49.0
        rm.register_open(state, fill)

        exits = rm.check_exits(state, "10:30", {"A": 48.5})  # below stop
        assert len(exits) == 1
        _, exit_price, reason = exits[0]
        assert reason == "stop"
        assert exit_price == 49.0

    def test_target_triggered_long(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        fill = _make_fill("A", Direction.LONG, entry=50.0, shares=100)
        fill.target_price = 52.0
        rm.register_open(state, fill)

        exits = rm.check_exits(state, "10:30", {"A": 52.5})  # above target
        assert len(exits) == 1
        _, _, reason = exits[0]
        assert reason == "target"

    def test_stop_triggered_short(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        fill = _make_fill("A", Direction.SHORT, entry=50.0, shares=100)
        fill.stop_price = 51.0
        rm.register_open(state, fill)

        exits = rm.check_exits(state, "10:30", {"A": 51.5})  # above stop
        assert len(exits) == 1
        _, _, reason = exits[0]
        assert reason == "stop"

    def test_eod_forces_exit(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        fill = _make_fill("A", Direction.LONG, entry=50.0, shares=100)
        fill.stop_price = 49.0
        fill.target_price = 54.0
        rm.register_open(state, fill)

        # Price is between stop and target but it's EOD
        exits = rm.check_exits(state, "15:55", {"A": 51.0})
        assert len(exits) == 1
        _, _, reason = exits[0]
        assert reason == "time_stop"

    def test_no_exit_when_price_between_stop_and_target(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        fill = _make_fill("A", Direction.LONG, entry=50.0, shares=100)
        fill.stop_price = 49.0
        fill.target_price = 54.0
        rm.register_open(state, fill)

        exits = rm.check_exits(state, "10:30", {"A": 51.5})
        assert len(exits) == 0


class TestCircuitBreaker:
    def test_no_trigger_below_limit(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        state.daily_pnl = -1500.0  # 1.5% < 2% limit
        triggered = rm.check_circuit_breaker(state)
        assert triggered is False
        assert state.circuit_breaker_triggered is False

    def test_trigger_at_limit(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        state.daily_pnl = -2000.0  # exactly 2%
        triggered = rm.check_circuit_breaker(state)
        assert triggered is True

    def test_trigger_prevents_new_entries(self):
        rm = RiskManager()
        state = RiskState("2024-01-10", 100_000.0, 100_000.0)
        state.daily_pnl = -3000.0
        rm.check_circuit_breaker(state)
        ok, reason = rm.can_enter(state)
        assert ok is False
        assert reason == "circuit_breaker"
