"""Short squeeze risk computation."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.config import AppSettings


@dataclass
class SqueezeRiskResult:
    score: float
    tradable: bool


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_squeeze_risk(settings: AppSettings, short_interest_row, float_shares: float | None, crowding_record: dict[str, object], premarket_metrics) -> SqueezeRiskResult:
    signal_settings = settings.signals.squeeze_risk
    short_pct_float = 0.0 if short_interest_row is None or short_interest_row["short_pct_float"] is None else float(short_interest_row["short_pct_float"])
    days_to_cover = 0.0 if short_interest_row is None or short_interest_row["days_to_cover"] is None else float(short_interest_row["days_to_cover"])
    float_score = 0.0 if not float_shares else _clamp(1.0 - min(float_shares / signal_settings.low_float_cap, 1.0))
    attention_score = 0.0 if crowding_record.get("attention_score") is None else float(crowding_record["attention_score"])
    gap_score = 0.0 if premarket_metrics.gap_pct is None else _clamp(abs(float(premarket_metrics.gap_pct)) / signal_settings.gap_magnitude_cap)
    momentum_score = 0.0 if premarket_metrics.relative_volume is None else _clamp(float(premarket_metrics.relative_volume) / 5.0)

    score = _clamp(
        (
            _clamp(short_pct_float / signal_settings.short_pct_float_cap)
            + _clamp(days_to_cover / signal_settings.days_to_cover_cap)
            + float_score
            + _clamp(attention_score / signal_settings.upward_attention_cap)
            + gap_score
            + momentum_score
        )
        / 6.0
    )
    return SqueezeRiskResult(score=score, tradable=score <= signal_settings.block_threshold)
