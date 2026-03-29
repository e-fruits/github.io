"""
src/backtest/engine.py

Event-driven backtest engine.

Architecture
────────────
The engine processes trading days chronologically.  For each day:

  1. Load the pre-built scanner watchlist from scanner_watchlist.
  2. Load intraday price bars (minute-level from Polygon if available,
     else simulate from daily OHLCV using a synthetic bar generator).
  3. Replay bars in time order.  At each bar:
       a. Check exit conditions for open positions (stop, target, EOD).
       b. For watchlist tickers not yet entered, evaluate entry signals.
       c. Simulate fills via execution.py.
       d. Update the RiskState via risk_manager.py.
  4. At day-end: force-close all open positions at the closing bar price.
  5. Persist fills to backtest_trades and daily summary to backtest_daily.

The engine is deliberately self-contained: it reads price data from the
DB and writes results to the DB.  No global state.

Bar simulation (when minute data is unavailable)
────────────────────────────────────────────────
We generate synthetic minute bars from OHLCV using a path-simulation
approach: open at the day's open, move toward high or low first (randomly,
seeded deterministically per ticker+date), then close at the day's close.
This is a rough approximation; real minute data is strongly preferred.
"""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Generator

import pandas as pd

from src.backtest.execution import Direction, ExitReason, Fill, Order, simulate_fill, close_fill
from src.backtest.risk_manager import RiskManager, RiskState
from src.backtest.strategy import MechanicalStrategy, WatchlistEntry
from src.utils.config import settings
from src.utils.db import fetch_df, get_connection, get_db, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)

_BT = settings["backtest"]


# ── Synthetic bar generation ───────────────────────────────────────────────────

@dataclass
class Bar:
    time: str    # HH:MM
    open: float
    high: float
    low: float
    close: float
    volume: int


def _seeded_rng(ticker: str, date_str: str) -> random.Random:
    seed = int(hashlib.sha256(f"{ticker}{date_str}".encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def simulate_bars(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    ticker: str,
    date_str: str,
    n_bars: int = 78,  # ~6.5 hours × 60 min / 5 min bars
) -> list[Bar]:
    """
    Generate *n_bars* synthetic 5-minute bars consistent with the day's OHLCV.
    Uses a seeded RNG for reproducibility.
    """
    rng = _seeded_rng(ticker, date_str)
    times = [f"{9 + (i * 5) // 60:02d}:{(30 + (i * 5)) % 60:02d}" for i in range(n_bars)]

    # Decide if price goes to high first or low first
    high_first = rng.random() > 0.5

    # Simple linear interpolation with noise
    bars: list[Bar] = []
    current = open_
    vol_per_bar = volume // n_bars

    for i, t in enumerate(times):
        progress = i / (n_bars - 1)
        if high_first:
            if progress < 0.5:
                target = open_ + (high - open_) * (progress / 0.5)
            else:
                target = high + (close - high) * ((progress - 0.5) / 0.5)
        else:
            if progress < 0.5:
                target = open_ + (low - open_) * (progress / 0.5)
            else:
                target = low + (close - low) * ((progress - 0.5) / 0.5)

        noise = rng.gauss(0, (high - low) * 0.05)
        bar_open  = current
        bar_close = max(low, min(high, target + noise))
        bar_high  = max(bar_open, bar_close) * (1 + abs(rng.gauss(0, 0.001)))
        bar_low   = min(bar_open, bar_close) * (1 - abs(rng.gauss(0, 0.001)))
        bar_high  = min(bar_high, high)
        bar_low   = max(bar_low, low)

        bars.append(Bar(
            time=t,
            open=round(bar_open, 4),
            high=round(bar_high, 4),
            low=round(bar_low, 4),
            close=round(bar_close, 4),
            volume=vol_per_bar,
        ))
        current = bar_close

    # Ensure last bar closes at the day's close
    bars[-1].close = close
    return bars


# ── Engine ─────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtest engine.

    Usage
    -----
    engine = BacktestEngine()
    engine.run(from_date="2022-01-01", to_date="2022-12-31")
    """

    def __init__(
        self,
        strategy=None,
        db_path=None,
        run_id: str | None = None,
    ) -> None:
        self.strategy    = strategy or MechanicalStrategy()
        self.risk_mgr    = RiskManager()
        self.db_path     = db_path
        self.run_id      = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self._starting_capital = float(_BT.get("starting_capital", 100_000))

    # ── Data loaders ────────────────────────────────────────────────────────

    def _load_watchlist(self, date_str: str) -> list[WatchlistEntry]:
        conn = get_connection(self.db_path)
        df = fetch_df(
            conn,
            "SELECT * FROM scanner_watchlist WHERE date = ?",
            (date_str,),
        )
        if df.empty:
            return []
        entries = []
        for _, row in df.iterrows():
            # Load pre-market data from daily_prices
            price_row = conn.execute(
                "SELECT pre_market_high, pre_market_low, open FROM daily_prices WHERE date = ? AND ticker = ?",
                (date_str, row["ticker"]),
            ).fetchone()
            entries.append(WatchlistEntry(
                ticker=str(row["ticker"]),
                date=date_str,
                catalyst_type=str(row.get("catalyst_type", "")),
                gap_pct=float(row.get("gap_pct") or 0.0),
                relative_volume=float(row.get("relative_volume") or 0.0),
                herd_score=float(row.get("herd_score") or 0.0),
                herd_direction=str(row.get("herd_direction", "neutral")),
                herd_stage=str(row.get("herd_stage", "forming")),
                market_cap=float(row.get("market_cap") or 0.0),
                pre_market_high=float(price_row["pre_market_high"]) if price_row and price_row["pre_market_high"] else None,
                pre_market_low=float(price_row["pre_market_low"])   if price_row and price_row["pre_market_low"]  else None,
                open_price=float(price_row["open"])                  if price_row and price_row["open"]           else None,
            ))
        return entries

    def _load_bars(self, ticker: str, date_str: str) -> list[Bar]:
        """
        Attempt to load minute bars from the DB.
        Falls back to synthetic bars from daily OHLCV.
        """
        # TODO: when minute data is ingested, query it here.
        # For now, always synthesise.
        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT open, high, low, close, volume FROM daily_prices WHERE date = ? AND ticker = ?",
            (date_str, ticker),
        ).fetchone()
        if row is None or row["open"] is None:
            return []
        return simulate_bars(
            open_=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"] or 0),
            ticker=ticker,
            date_str=date_str,
        )

    # ── Single-day simulation ────────────────────────────────────────────────

    def _simulate_day(self, date_str: str, state: RiskState) -> list[Fill]:
        entries      = self._load_watchlist(date_str)
        all_tickers  = [e.ticker for e in entries]
        entry_map    = {e.ticker: e for e in entries}
        entered      = set()
        closed_fills: list[Fill] = []

        # Preload all bars
        bars_by_ticker: dict[str, list[Bar]] = {
            t: self._load_bars(t, date_str) for t in all_tickers
        }

        # Collect all unique times across all tickers
        all_times = sorted({b.time for bars in bars_by_ticker.values() for b in bars})

        for t_str in all_times:
            price_snapshot = {
                ticker: next(
                    (b.close for b in bars if b.time == t_str), None
                )
                for ticker, bars in bars_by_ticker.items()
            }

            # 1. Check exits for open positions
            exits = self.risk_mgr.check_exits(state, t_str, {k: v for k, v in price_snapshot.items() if v})
            for fill, exit_price, reason in exits:
                close_fill(fill, exit_price, t_str, ExitReason(reason))
                self.risk_mgr.register_close(state, fill)
                closed_fills.append(fill)

            # 2. Evaluate entry signals
            if not state.circuit_breaker_triggered:
                for ticker, entry in entry_map.items():
                    if ticker in entered:
                        continue
                    price = price_snapshot.get(ticker)
                    if price is None:
                        continue
                    signal = self.strategy.signal_for_entry(entry, price, t_str)
                    if signal.signal_type == "none":
                        continue
                    # Breakout entry: only enter when price crosses breakout level
                    if signal.entry_trigger == "breakout" and signal.breakout_level:
                        if price < signal.breakout_level:
                            continue

                    order = self.strategy.size_order(signal, state, self.risk_mgr, price)
                    if order is None:
                        continue

                    avg_vol = self._get_avg_volume(ticker, date_str)
                    fill = simulate_fill(order, price, avg_vol)
                    self.risk_mgr.register_open(state, fill)
                    entered.add(ticker)

        # EOD: force-close anything still open
        for fill in list(state.open_positions):
            last_price = self._get_closing_price(fill.ticker, date_str)
            if last_price:
                close_fill(fill, last_price, "15:55:00", ExitReason.TIME_STOP)
                self.risk_mgr.register_close(state, fill)
                closed_fills.append(fill)

        return closed_fills

    def _get_avg_volume(self, ticker: str, date_str: str) -> float:
        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT avg_volume_20d FROM daily_prices WHERE date = ? AND ticker = ?",
            (date_str, ticker),
        ).fetchone()
        return float(row["avg_volume_20d"]) if row and row["avg_volume_20d"] else 0.0

    def _get_closing_price(self, ticker: str, date_str: str) -> float | None:
        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT close FROM daily_prices WHERE date = ? AND ticker = ?",
            (date_str, ticker),
        ).fetchone()
        return float(row["close"]) if row and row["close"] else None

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist_day(self, state: RiskState, fills: list[Fill]) -> None:
        # Trade rows
        trade_rows = []
        for f in fills:
            trade_rows.append({
                "run_id": self.run_id,
                "date": f.date,
                "ticker": f.ticker,
                "entry_time": f.entry_time,
                "exit_time": f.exit_time,
                "direction": f.direction.value,
                "entry_price": f.entry_price,
                "exit_price": f.exit_price,
                "position_type": f.position_type.value,
                "leverage_multiple": f.leverage_multiple,
                "shares_or_contracts": f.shares,
                "pnl_dollars": f.pnl_dollars,
                "pnl_pct": f.pnl_pct,
                "stop_price": f.stop_price,
                "target_price": f.target_price,
                "exit_reason": f.exit_reason.value,
                "herd_score_at_entry": f.herd_score_at_entry,
                "catalyst_type": f.catalyst_type,
                "strategy_signal": f.strategy_signal,
            })

        wins   = sum(1 for f in fills if f.pnl_dollars > 0)
        losses = sum(1 for f in fills if f.pnl_dollars <= 0)

        daily_row = {
            "run_id": self.run_id,
            "date": state.date,
            "starting_capital": state.starting_capital,
            "ending_capital": state.current_capital,
            "daily_pnl": state.daily_pnl,
            "daily_return_pct": state.daily_return_pct,
            "num_trades": len(fills),
            "num_wins": wins,
            "num_losses": losses,
            "max_drawdown_intraday": state.max_drawdown_intraday,
            "positions_open": state.num_open,
        }

        with get_db(self.db_path) as conn:
            if trade_rows:
                upsert_rows(conn, "backtest_trades", trade_rows)
            upsert_rows(conn, "backtest_daily", [daily_row])

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self, from_date: str, to_date: str | None = None) -> dict:
        """
        Run the full backtest over [from_date, to_date].
        Returns a metrics dict (see metrics.py).
        """
        from src.backtest.metrics import compute_metrics

        start = date.fromisoformat(from_date)
        end   = date.fromisoformat(to_date) if to_date else date.today()

        capital = self._starting_capital
        log.info("backtest_start", run_id=self.run_id, from_date=from_date, to_date=str(end), capital=capital)

        current = start
        while current <= end:
            if current.weekday() < 5:
                ds = current.isoformat()
                state = RiskState(
                    date=ds,
                    starting_capital=capital,
                    current_capital=capital,
                )
                try:
                    fills = self._simulate_day(ds, state)
                    self._persist_day(state, fills)
                    capital = state.current_capital
                    log.info(
                        "day_complete",
                        date=ds,
                        trades=len(fills),
                        pnl=round(state.daily_pnl, 2),
                        capital=round(capital, 2),
                    )
                except Exception as exc:
                    log.error("day_failed", date=ds, error=str(exc))
            current += timedelta(days=1)

        log.info("backtest_complete", run_id=self.run_id, final_capital=round(capital, 2))

        # Compute and return metrics
        conn = get_connection(self.db_path)
        daily_df  = fetch_df(conn, "SELECT * FROM backtest_daily  WHERE run_id = ? ORDER BY date", (self.run_id,))
        trades_df = fetch_df(conn, "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY date", (self.run_id,))
        return compute_metrics(daily_df, trades_df, self._starting_capital)
