"""Premarket dislocation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.config import AppSettings


@dataclass
class PremarketMetrics:
    gap_pct: float | None
    relative_volume: float | None
    dollar_volume: float | None
    spread_estimate_pct: float | None
    meaningful_dislocation: bool
    sufficient_liquidity: bool


def compute_premarket_metrics(settings: AppSettings, daily_price_row) -> PremarketMetrics:
    if daily_price_row is None:
        return PremarketMetrics(None, None, None, None, False, False)

    gap_pct = daily_price_row["gap_pct"]
    relative_volume = daily_price_row["pre_market_relative_volume"]
    dollar_volume = daily_price_row["pre_market_dollar_volume"]
    pre_high = daily_price_row["pre_market_high"]
    pre_low = daily_price_row["pre_market_low"]
    spread_estimate_pct = None
    if pre_high is not None and pre_low is not None and (pre_high + pre_low) != 0:
        midpoint = (float(pre_high) + float(pre_low)) / 2.0
        spread_estimate_pct = (float(pre_high) - float(pre_low)) / midpoint

    thresholds = settings.signals.premarket
    meaningful_dislocation = bool(gap_pct is not None and abs(float(gap_pct)) >= thresholds.meaningful_gap_pct)
    sufficient_liquidity = bool(
        dollar_volume is not None
        and float(dollar_volume) >= thresholds.min_dollar_volume
        and (relative_volume is None or float(relative_volume) >= thresholds.min_relative_volume)
        and (spread_estimate_pct is None or spread_estimate_pct <= thresholds.max_spread_estimate_pct)
    )
    return PremarketMetrics(
        gap_pct=None if gap_pct is None else float(gap_pct),
        relative_volume=None if relative_volume is None else float(relative_volume),
        dollar_volume=None if dollar_volume is None else float(dollar_volume),
        spread_estimate_pct=spread_estimate_pct,
        meaningful_dislocation=meaningful_dislocation,
        sufficient_liquidity=sufficient_liquidity,
    )
