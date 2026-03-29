"""
tests/test_backtest.py

Unit tests for the backtest engine, execution, and metrics.
No API calls; uses synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.backtest.execution import Direction, ExitReason, Fill, Order, PositionType, close_fill, simulate_fill
from src.backtest.metrics import compute_metrics
from src.utils.db import get_connection, get_db, init_db, upsert_rows

import pandas as pd


# ── Execution tests ────────────────────────────────────────────────────────────

class TestExecution:
    def _make_order(self, direction=Direction.LONG, shares=100, price=50.0) -> Order:
        return Order(
            ticker="TEST",
            date="2024-01-10",
            time="09:30:00",
            direction=direction,
            shares=shares,
            position_type=PositionType.EQUITY,
            leverage_multiple=1.0,
            stop_price=49.0 if direction == Direction.LONG else 51.0,
            target_price=52.0 if direction == Direction.LONG else 48.0,
        )

    def test_fill_long_has_slippage(self):
        order = self._make_order(Direction.LONG)
        fill = simulate_fill(order, market_price=50.0)
        # Long fill should be above market price (slippage against us)
        assert fill.entry_price > 50.0

    def test_fill_short_has_slippage(self):
        order = self._make_order(Direction.SHORT)
        fill = simulate_fill(order, market_price=50.0)
        # Short fill should be below market price
        assert fill.entry_price < 50.0

    def test_fill_has_commission(self):
        order = self._make_order(shares=100)
        fill = simulate_fill(order, market_price=50.0)
        assert fill.commission > 0

    def test_close_long_win(self):
        order = self._make_order(Direction.LONG, shares=100)
        fill = simulate_fill(order, market_price=50.0)
        close_fill(fill, exit_price=55.0, exit_time="14:00:00", exit_reason=ExitReason.TARGET)
        assert fill.pnl_dollars > 0
        assert fill.exit_reason == ExitReason.TARGET

    def test_close_long_loss(self):
        order = self._make_order(Direction.LONG, shares=100)
        fill = simulate_fill(order, market_price=50.0)
        close_fill(fill, exit_price=48.0, exit_time="10:30:00", exit_reason=ExitReason.STOP)
        assert fill.pnl_dollars < 0
        assert fill.exit_reason == ExitReason.STOP

    def test_close_short_win(self):
        order = self._make_order(Direction.SHORT, shares=100)
        fill = simulate_fill(order, market_price=50.0)
        close_fill(fill, exit_price=46.0, exit_time="11:00:00", exit_reason=ExitReason.TARGET)
        assert fill.pnl_dollars > 0

    def test_pnl_pct_sign(self):
        order = self._make_order(Direction.LONG, shares=200)
        fill = simulate_fill(order, market_price=100.0)
        close_fill(fill, 110.0, "13:00:00", ExitReason.TARGET)
        assert fill.pnl_pct > 0.0


# ── Risk manager tests ─────────────────────────────────────────────────────────

class TestRiskManager:
    def _make_state(self, capital=100_000.0):
        from src.backtest.risk_manager import RiskState
        return RiskState(date="2024-01-10", starting_capital=capital, current_capital=capital)

    def test_can_enter_clean(self):
        from src.backtest.risk_manager import RiskManager
        rm = RiskManager()
        state = self._make_state()
        ok, reason = rm.can_enter(state)
        assert ok is True
        assert reason == ""

    def test_circuit_breaker_blocks_entry(self):
        from src.backtest.risk_manager import RiskManager
        rm = RiskManager()
        state = self._make_state()
        state.circuit_breaker_triggered = True
        ok, reason = rm.can_enter(state)
        assert ok is False

    def test_circuit_breaker_triggers_on_loss(self):
        from src.backtest.risk_manager import RiskManager, RiskState
        rm = RiskManager()
        state = RiskState("2024-01-10", starting_capital=100_000.0, current_capital=100_000.0)
        state.daily_pnl = -2500.0   # 2.5% loss > 2% limit
        triggered = rm.check_circuit_breaker(state)
        assert triggered is True
        assert state.circuit_breaker_triggered is True

    def test_max_positions_blocks_entry(self):
        from src.backtest.risk_manager import RiskManager
        from src.backtest.execution import Order, Direction, PositionType, simulate_fill
        rm = RiskManager()
        state = self._make_state()
        # Fill up all position slots
        for i in range(5):
            order = Order(f"T{i}", "2024-01-10", "09:30:00", Direction.LONG, 100, position_type=PositionType.EQUITY, leverage_multiple=1.0)
            fill = simulate_fill(order, 50.0)
            rm.register_open(state, fill)
        ok, reason = rm.can_enter(state)
        assert ok is False
        assert "max_positions" in reason

    def test_stop_price_long(self):
        from src.backtest.risk_manager import RiskManager
        rm = RiskManager()
        stop = rm.calculate_stop(100.0, Direction.LONG)
        assert stop < 100.0

    def test_stop_price_short(self):
        from src.backtest.risk_manager import RiskManager
        rm = RiskManager()
        stop = rm.calculate_stop(100.0, Direction.SHORT)
        assert stop > 100.0

    def test_eod_detection(self):
        from src.backtest.risk_manager import RiskManager
        rm = RiskManager()
        assert rm.is_eod("15:55") is True
        assert rm.is_eod("15:56") is True
        assert rm.is_eod("15:54") is False
        assert rm.is_eod("09:30") is False


# ── Strategy tests ─────────────────────────────────────────────────────────────

class TestStrategy:
    def _make_entry(self, herd_score=6.5, herd_stage="saturated", gap=0.05):
        from src.backtest.strategy import WatchlistEntry
        return WatchlistEntry(
            ticker="MEME", date="2024-01-10", catalyst_type="earnings",
            gap_pct=gap, relative_volume=5.0, herd_score=herd_score,
            herd_direction="bullish", herd_stage=herd_stage, market_cap=1e9,
            pre_market_high=52.0, pre_market_low=48.0, open_price=51.0,
        )

    def test_fade_signal_on_saturated(self):
        from src.backtest.strategy import MechanicalStrategy, SIGNAL_FADE
        strat = MechanicalStrategy()
        entry = self._make_entry(herd_score=7.0, herd_stage="saturated", gap=0.06)
        signal = strat.signal_for_entry(entry, current_price=51.0, current_time="09:30:00")
        assert signal.signal_type == SIGNAL_FADE
        assert signal.direction == Direction.SHORT

    def test_follow_signal_on_forming(self):
        from src.backtest.strategy import MechanicalStrategy, SIGNAL_FOLLOW
        strat = MechanicalStrategy()
        entry = self._make_entry(herd_score=5.0, herd_stage="forming", gap=0.04)
        signal = strat.signal_for_entry(entry, current_price=51.0, current_time="09:30:00")
        assert signal.signal_type == SIGNAL_FOLLOW
        assert signal.direction == Direction.LONG

    def test_no_signal_low_herd_score(self):
        from src.backtest.strategy import MechanicalStrategy, SIGNAL_NONE
        strat = MechanicalStrategy()
        entry = self._make_entry(herd_score=1.0, herd_stage="forming", gap=0.05)
        signal = strat.signal_for_entry(entry, current_price=51.0, current_time="09:30:00")
        assert signal.signal_type == SIGNAL_NONE

    def test_no_signal_small_gap(self):
        from src.backtest.strategy import MechanicalStrategy, SIGNAL_NONE
        strat = MechanicalStrategy()
        entry = self._make_entry(herd_score=7.0, herd_stage="forming", gap=0.005)
        signal = strat.signal_for_entry(entry, current_price=51.0, current_time="09:30:00")
        assert signal.signal_type == SIGNAL_NONE


# ── Metrics tests ──────────────────────────────────────────────────────────────

class TestMetrics:
    def _make_daily(self, days=10, daily_return=0.001) -> pd.DataFrame:
        import numpy as np
        dates = pd.date_range("2024-01-02", periods=days, freq="B").strftime("%Y-%m-%d")
        capital = 100_000.0
        records = []
        for i, d in enumerate(dates):
            pnl = capital * daily_return
            records.append({
                "date": d,
                "starting_capital": capital,
                "ending_capital": capital + pnl,
                "daily_pnl": pnl,
                "daily_return_pct": daily_return,
                "num_trades": 2,
                "num_wins": 1,
                "num_losses": 1,
                "max_drawdown_intraday": 0.0,
                "positions_open": 0,
            })
            capital += pnl
        return pd.DataFrame(records)

    def _make_trades(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": "2024-01-02", "ticker": "A", "pnl_dollars": 200.0, "pnl_pct": 0.02,
             "exit_reason": "target", "catalyst_type": "earnings",
             "entry_time": "09:30:00", "exit_time": "11:30:00"},
            {"date": "2024-01-03", "ticker": "B", "pnl_dollars": -100.0, "pnl_pct": -0.01,
             "exit_reason": "stop", "catalyst_type": "earnings",
             "entry_time": "09:35:00", "exit_time": "10:00:00"},
        ])

    def test_total_return_positive(self):
        daily = self._make_daily(days=20, daily_return=0.001)
        trades = self._make_trades()
        m = compute_metrics(daily, trades, 100_000.0)
        assert m["total_return_pct"] > 0

    def test_win_rate_calculation(self):
        daily = self._make_daily(days=5)
        trades = self._make_trades()
        m = compute_metrics(daily, trades, 100_000.0)
        assert m["win_rate_pct"] == pytest.approx(50.0)

    def test_sharpe_positive_for_positive_returns(self):
        daily = self._make_daily(days=100, daily_return=0.001)
        m = compute_metrics(daily, pd.DataFrame(), 100_000.0)
        assert m["sharpe_ratio"] > 0

    def test_max_drawdown_non_positive(self):
        daily = self._make_daily(days=20, daily_return=-0.001)
        m = compute_metrics(daily, pd.DataFrame(), 100_000.0)
        assert m["max_drawdown_pct"] <= 0

    def test_empty_trades_no_error(self):
        daily = self._make_daily(days=5)
        m = compute_metrics(daily, pd.DataFrame(), 100_000.0)
        assert "total_trades" in m
        assert m["total_trades"] == 0
