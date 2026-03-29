"""
src/backtest/metrics.py

Performance analytics computed from backtest_trades and backtest_daily.

Metrics
───────
  Total return, CAGR
  Sharpe ratio (annualised, risk-free = 0)
  Sortino ratio (annualised, using downside deviation)
  Max drawdown (from equity curve)
  Win rate, average win, average loss, profit factor
  Average holding time
  Trades per day
  Expectancy ($ per trade)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    starting_capital: float,
) -> dict[str, Any]:
    """
    Compute full performance metrics.

    Parameters
    ----------
    daily_df  : backtest_daily rows for a single run_id, sorted by date.
    trades_df : backtest_trades rows for the same run_id.
    starting_capital : initial account size.

    Returns a dict suitable for pretty-printing or JSON serialisation.
    """
    if daily_df.empty:
        return {"error": "no daily data"}

    daily_df = daily_df.sort_values("date").copy()
    daily_df["daily_return_pct"] = pd.to_numeric(daily_df["daily_return_pct"], errors="coerce").fillna(0.0)
    equity = daily_df["ending_capital"].astype(float)

    total_days    = len(daily_df)
    years         = total_days / _TRADING_DAYS_PER_YEAR
    final_capital = float(equity.iloc[-1]) if not equity.empty else starting_capital

    # ── Return metrics ──────────────────────────────────────────────────────
    total_return_pct = (final_capital - starting_capital) / starting_capital * 100
    cagr = ((final_capital / starting_capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    returns = daily_df["daily_return_pct"].values

    # ── Risk metrics ────────────────────────────────────────────────────────
    mean_return = float(np.mean(returns))
    std_return  = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    sharpe = (mean_return / std_return * math.sqrt(_TRADING_DAYS_PER_YEAR)) if std_return > 0 else 0.0

    downside = returns[returns < 0]
    down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino  = (mean_return / down_std * math.sqrt(_TRADING_DAYS_PER_YEAR)) if down_std > 0 else 0.0

    # Max drawdown from equity curve
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown_pct = float(drawdown.min() * 100)

    # ── Trade-level metrics ─────────────────────────────────────────────────
    if trades_df.empty:
        trade_metrics: dict[str, Any] = {
            "total_trades": 0,
            "win_rate_pct": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "profit_factor": None,
            "expectancy_dollars": None,
            "avg_holding_bars": None,
            "trades_per_day": 0.0,
        }
    else:
        trades_df = trades_df.copy()
        trades_df["pnl_dollars"] = pd.to_numeric(trades_df["pnl_dollars"], errors="coerce").fillna(0.0)
        trades_df["pnl_pct"]     = pd.to_numeric(trades_df["pnl_pct"],     errors="coerce").fillna(0.0)

        wins  = trades_df[trades_df["pnl_dollars"] > 0]
        losses = trades_df[trades_df["pnl_dollars"] <= 0]

        win_rate  = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0.0
        avg_win   = float(wins["pnl_pct"].mean() * 100)   if not wins.empty   else 0.0
        avg_loss  = float(losses["pnl_pct"].mean() * 100) if not losses.empty else 0.0
        gross_win  = float(wins["pnl_dollars"].sum())
        gross_loss = abs(float(losses["pnl_dollars"].sum()))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        expectancy = float(trades_df["pnl_dollars"].mean())

        # Holding time: approximate from entry/exit time strings (HH:MM:SS)
        avg_holding: float | None = None
        if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
            try:
                def _hhmm_to_minutes(t: str) -> float:
                    parts = str(t).split(":")
                    return int(parts[0]) * 60 + int(parts[1])
                trades_df["holding_min"] = (
                    trades_df["exit_time"].apply(_hhmm_to_minutes)
                    - trades_df["entry_time"].apply(_hhmm_to_minutes)
                ).clip(lower=0)
                avg_holding = float(trades_df["holding_min"].mean())
            except Exception:
                pass

        trade_metrics = {
            "total_trades":       len(trades_df),
            "win_rate_pct":       round(win_rate, 2),
            "avg_win_pct":        round(avg_win, 4),
            "avg_loss_pct":       round(avg_loss, 4),
            "profit_factor":      round(profit_factor, 3),
            "expectancy_dollars": round(expectancy, 2),
            "avg_holding_minutes": round(avg_holding, 1) if avg_holding is not None else None,
            "trades_per_day":     round(len(trades_df) / max(total_days, 1), 2),
        }

        # Break down by exit reason
        if "exit_reason" in trades_df.columns:
            trade_metrics["exit_reason_breakdown"] = (
                trades_df.groupby("exit_reason")["pnl_dollars"]
                .agg(count="count", total_pnl="sum")
                .round(2)
                .to_dict("index")
            )

        # Break down by catalyst type
        if "catalyst_type" in trades_df.columns:
            trade_metrics["catalyst_breakdown"] = (
                trades_df.groupby("catalyst_type")["pnl_dollars"]
                .agg(count="count", win_rate=lambda x: (x > 0).mean() * 100, total_pnl="sum")
                .round(2)
                .to_dict("index")
            )

    return {
        "starting_capital": starting_capital,
        "ending_capital":   round(final_capital, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct":         round(cagr, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "sortino_ratio":    round(sortino, 3),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "trading_days":     total_days,
        **trade_metrics,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Pretty-print a metrics dict to stdout."""
    print("\n" + "=" * 52)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 52)
    simple_keys = [
        ("starting_capital",     "Starting Capital",     "${:,.0f}"),
        ("ending_capital",       "Ending Capital",       "${:,.0f}"),
        ("total_return_pct",     "Total Return",         "{:+.2f}%"),
        ("cagr_pct",             "CAGR",                 "{:+.2f}%"),
        ("sharpe_ratio",         "Sharpe Ratio",         "{:.3f}"),
        ("sortino_ratio",        "Sortino Ratio",        "{:.3f}"),
        ("max_drawdown_pct",     "Max Drawdown",         "{:.2f}%"),
        ("trading_days",         "Trading Days",         "{:,}"),
        ("total_trades",         "Total Trades",         "{:,}"),
        ("win_rate_pct",         "Win Rate",             "{:.1f}%"),
        ("avg_win_pct",          "Avg Win",              "{:+.2f}%"),
        ("avg_loss_pct",         "Avg Loss",             "{:+.2f}%"),
        ("profit_factor",        "Profit Factor",        "{:.3f}"),
        ("expectancy_dollars",   "Expectancy / Trade",   "${:+.2f}"),
        ("avg_holding_minutes",  "Avg Holding (min)",    "{:.0f} min"),
        ("trades_per_day",       "Trades / Day",         "{:.2f}"),
    ]
    for key, label, fmt in simple_keys:
        val = metrics.get(key)
        if val is None:
            continue
        try:
            formatted = fmt.format(val)
        except Exception:
            formatted = str(val)
        print(f"  {label:<28} {formatted:>12}")
    print("=" * 52)

    if "exit_reason_breakdown" in metrics:
        print("\n  Exit reason breakdown:")
        for reason, stats in metrics["exit_reason_breakdown"].items():
            print(f"    {reason:<20} {stats['count']:>4} trades  ${stats['total_pnl']:>10,.2f}")

    if "catalyst_breakdown" in metrics:
        print("\n  Catalyst breakdown:")
        for ctype, stats in metrics["catalyst_breakdown"].items():
            print(f"    {ctype:<24} {stats['count']:>4} trades  {stats['win_rate']:.1f}% win  ${stats['total_pnl']:>10,.2f}")

    print()


def equity_curve(daily_df: pd.DataFrame) -> pd.Series:
    """Return a Series of ending_capital indexed by date."""
    return daily_df.set_index("date")["ending_capital"].astype(float)
