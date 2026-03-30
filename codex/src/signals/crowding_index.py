"""Crowding index computation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any

from src.utils.config import AppSettings
from src.utils.db import DatabaseManager


@dataclass
class AttentionResult:
    label: str
    score: float
    components: dict[str, float | None]


@dataclass
class PositioningPressureResult:
    label: str
    score: float
    components: dict[str, float | None]


@dataclass
class CrowdTrapRiskResult:
    label: str
    score: float
    components: dict[str, float | None]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize(value: float | None, cap: float) -> float | None:
    if value is None:
        return None
    return _clamp(value / cap) if cap else 0.0


def _categorize(score: float, medium_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def compute_attention(settings: AppSettings, wsb_rows: list[Any], options_row: Any) -> AttentionResult:
    signal_settings = settings.signals.attention
    mention_velocity = _average([row["mention_velocity_pct"] for row in wsb_rows if row["mention_velocity_pct"] is not None])
    mention_vs_baseline = _average([row["mention_vs_baseline"] for row in wsb_rows if row["mention_vs_baseline"] is not None])
    unusual_options = None if options_row is None else options_row["unusual_volume_ratio"]

    velocity_score = _normalize(mention_velocity, signal_settings.mention_velocity_cap)
    baseline_score = _normalize(mention_vs_baseline, signal_settings.mention_vs_baseline_cap)
    wsb_score = _average([value for value in [velocity_score, baseline_score] if value is not None]) or 0.0
    options_score = _normalize(unusual_options, signal_settings.unusual_options_volume_cap)

    if options_score is None:
        score = _clamp(wsb_score)
    else:
        score = _clamp(signal_settings.wsb_weight * wsb_score + signal_settings.options_weight * options_score)

    return AttentionResult(
        label=_categorize(score, signal_settings.medium_threshold, signal_settings.high_threshold),
        score=score,
        components={
            "mention_velocity_pct": mention_velocity,
            "mention_vs_baseline": mention_vs_baseline,
            "unusual_options_volume_ratio": unusual_options,
        },
    )


def compute_positioning_pressure(settings: AppSettings, options_row: Any, short_interest_row: Any) -> PositioningPressureResult:
    signal_settings = settings.signals.positioning
    small_lot_ratio = None
    unusual_options = None
    if options_row is not None:
        small_lot_call = options_row["small_lot_call_volume"] or 0.0
        small_lot_put = options_row["small_lot_put_volume"] or 0.0
        if small_lot_put not in (None, 0, 0.0):
            small_lot_ratio = float(small_lot_call) / float(small_lot_put)
        unusual_options = options_row["unusual_volume_ratio"]

    short_pct_float = None if short_interest_row is None else short_interest_row["short_pct_float"]
    days_to_cover = None if short_interest_row is None else short_interest_row["days_to_cover"]

    option_subscore = _average(
        [
            value
            for value in [
                _normalize(small_lot_ratio, signal_settings.small_lot_call_put_ratio_cap),
                _normalize(unusual_options, signal_settings.unusual_options_volume_cap),
            ]
            if value is not None
        ]
    )
    short_subscore = _average(
        [
            value
            for value in [
                _normalize(short_pct_float, signal_settings.short_pct_float_cap),
                _normalize(days_to_cover, signal_settings.days_to_cover_cap),
            ]
            if value is not None
        ]
    )

    if option_subscore is None and short_subscore is None:
        score = 0.0
    elif option_subscore is None:
        score = short_subscore or 0.0
    elif short_subscore is None:
        score = option_subscore
    else:
        score = _clamp((option_subscore + short_subscore) / 2.0)

    return PositioningPressureResult(
        label=_categorize(score, signal_settings.medium_threshold, signal_settings.high_threshold),
        score=score,
        components={
            "small_lot_call_put_ratio": small_lot_ratio,
            "unusual_options_volume_ratio": unusual_options,
            "short_pct_float": short_pct_float,
            "days_to_cover": days_to_cover,
        },
    )


def compute_crowd_trap_risk(
    settings: AppSettings,
    attention: AttentionResult,
    positioning: PositioningPressureResult,
    price_context: dict[str, float | bool | None],
) -> CrowdTrapRiskResult:
    trap_settings = settings.signals.trap_risk
    level_components = [
        1.0 if bool(price_context.get("near_round_number")) else 0.0,
        1.0 if bool(price_context.get("near_52w_extreme")) else 0.0,
        1.0 if bool(price_context.get("near_support_resistance")) else 0.0,
    ]
    level_score = sum(level_components) / len(level_components)
    score = _clamp(((attention.score + positioning.score) / 2.0) * level_score)
    label = "present" if score >= trap_settings.present_threshold and attention.label == "high" and positioning.label == "high" else "absent"
    return CrowdTrapRiskResult(
        label=label,
        score=score,
        components={
            "attention_score": attention.score,
            "positioning_score": positioning.score,
            "near_round_number": 1.0 if bool(price_context.get("near_round_number")) else 0.0,
            "near_52w_extreme": 1.0 if bool(price_context.get("near_52w_extreme")) else 0.0,
            "near_support_resistance": 1.0 if bool(price_context.get("near_support_resistance")) else 0.0,
        },
    )


def compute_price_context(settings: AppSettings, current_price: float | None, history_rows: list[Any]) -> dict[str, bool | float | None]:
    if current_price is None or not history_rows:
        return {
            "near_round_number": False,
            "near_52w_extreme": False,
            "near_support_resistance": False,
        }

    trap_settings = settings.signals.trap_risk
    closes = [float(row["close"]) for row in history_rows if row["close"] is not None]
    highs = [float(row["high"]) for row in history_rows if row["high"] is not None]
    lows = [float(row["low"]) for row in history_rows if row["low"] is not None]
    recent_closes = closes[:20]

    near_round_number = any(abs(current_price - level) / level <= trap_settings.round_number_band_pct for level in trap_settings.round_numbers)
    near_52w_extreme = False
    if highs and lows:
        near_high = abs(current_price - max(highs)) / max(highs) <= trap_settings.yearly_extreme_band_pct
        near_low = abs(current_price - min(lows)) / max(min(lows), 0.01) <= trap_settings.yearly_extreme_band_pct
        near_52w_extreme = near_high or near_low

    near_support_resistance = False
    if recent_closes:
        support = min(recent_closes)
        resistance = max(recent_closes)
        near_support = abs(current_price - support) / max(support, 0.01) <= trap_settings.support_resistance_band_pct
        near_resistance = abs(current_price - resistance) / max(resistance, 0.01) <= trap_settings.support_resistance_band_pct
        near_support_resistance = near_support or near_resistance

    return {
        "near_round_number": near_round_number,
        "near_52w_extreme": near_52w_extreme,
        "near_support_resistance": near_support_resistance,
    }


def build_crowding_index_record(settings: AppSettings, db: DatabaseManager, ticker: str, trade_date: date) -> dict[str, object]:
    trade_date_str = trade_date.isoformat()
    wsb_rows = db.fetch_wsb_rows(ticker, trade_date_str)
    options_row = db.fetch_options_flow_row(ticker, trade_date_str)
    decision_time = datetime.combine(trade_date, time(hour=23, minute=59), tzinfo=timezone.utc)
    short_interest_row = db.fetch_latest_short_interest_as_of(ticker, decision_time)
    daily_price_row = db.fetch_row_by_date("daily_prices", ticker, trade_date_str)
    history_rows = db.fetch_price_history(ticker, trade_date_str, limit=252)

    current_reference_price = None if daily_price_row is None else (daily_price_row["pre_market_high"] or daily_price_row["open"] or daily_price_row["close"])
    price_context = compute_price_context(settings, current_reference_price, history_rows)
    attention = compute_attention(settings, wsb_rows, options_row)
    positioning = compute_positioning_pressure(settings, options_row, short_interest_row)
    trap = compute_crowd_trap_risk(settings, attention, positioning, price_context)
    component_details = {
        "attention": attention.components,
        "positioning": positioning.components,
        "trap": trap.components,
        "price_context": price_context,
    }
    as_of_candidates = []
    for row in list(wsb_rows) + [options_row, short_interest_row, daily_price_row]:
        if row is not None and "as_of_timestamp" in row.keys():
            as_of_candidates.append(str(row["as_of_timestamp"]))
    as_of_timestamp = max(as_of_candidates) if as_of_candidates else datetime.combine(trade_date, time(hour=13), tzinfo=timezone.utc).isoformat()

    return {
        "date": trade_date_str,
        "ticker": ticker,
        "attention": attention.label,
        "positioning_pressure": positioning.label,
        "crowd_trap_risk": trap.label,
        "attention_score": attention.score,
        "positioning_score": positioning.score,
        "trap_score": trap.score,
        "component_details": json.dumps(component_details, sort_keys=True),
        "as_of_timestamp": as_of_timestamp,
    }


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
