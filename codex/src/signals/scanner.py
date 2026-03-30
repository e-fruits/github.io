"""Historical watchlist generation."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path

from src.pipeline.short_interest import ShortInterestLoader, StubShortInterestClient
from src.signals.crowding_index import build_crowding_index_record
from src.signals.premarket_metrics import compute_premarket_metrics
from src.signals.squeeze_risk import compute_squeeze_risk
from src.utils.config import AppSettings, load_settings
from src.utils.db import DatabaseManager


def classify_setup(settings: AppSettings, catalyst_direction: str, crowding_record: dict[str, object], premarket_metrics, squeeze_risk_result) -> tuple[str, float]:
    attention_score = float(crowding_record["attention_score"])
    positioning_score = float(crowding_record["positioning_score"])
    trap_score = float(crowding_record["trap_score"])
    base_score = (attention_score + positioning_score + (1.0 - trap_score)) / 3.0

    if catalyst_direction == "positive":
        if base_score >= settings.signals.scanner.continuation_score_threshold and crowding_record["crowd_trap_risk"] == "absent":
            return "long_continuation", base_score
        if base_score >= settings.signals.scanner.mean_reversion_score_threshold:
            return "long_mean_reversion", base_score
        return "skip", base_score

    if catalyst_direction == "negative":
        if squeeze_risk_result.tradable and base_score >= settings.signals.scanner.continuation_score_threshold:
            return "short_continuation", base_score
        if squeeze_risk_result.tradable and base_score >= settings.signals.scanner.mean_reversion_score_threshold:
            return "short_mean_reversion", base_score
        return "skip", base_score

    return "skip", base_score


class HistoricalScanner:
    def __init__(self, settings: AppSettings, db: DatabaseManager) -> None:
        self.settings = settings
        self.db = db
        self.short_interest_loader = ShortInterestLoader(
            settings=settings,
            db=db,
            client=StubShortInterestClient(settings.short_interest.report_publication_lag_business_days),
        )

    def scan_day(self, trade_date: date) -> int:
        universe_rows = self.db.fetch_universe_rows(trade_date.isoformat())
        watchlist_rows: list[dict[str, object]] = []
        crowding_rows: list[dict[str, object]] = []
        for universe_row in universe_rows:
            ticker = universe_row["ticker"]
            catalyst_rows = self.db.fetch_catalyst_rows(ticker, trade_date.isoformat())
            if not catalyst_rows:
                continue

            daily_price_row = self.db.fetch_row_by_date("daily_prices", ticker, trade_date.isoformat())
            premarket = compute_premarket_metrics(self.settings, daily_price_row)
            if not (premarket.meaningful_dislocation and premarket.sufficient_liquidity):
                continue

            crowding_record = build_crowding_index_record(self.settings, self.db, ticker, trade_date)
            crowding_rows.append(crowding_record)
            latest_short_interest = self.short_interest_loader.latest_available_as_of(ticker, trade_date)
            squeeze_risk = compute_squeeze_risk(
                self.settings,
                latest_short_interest,
                None if latest_short_interest is None else latest_short_interest["float_shares"],
                crowding_record,
                premarket,
            )
            primary_catalyst = catalyst_rows[0]
            suggested_setup, setup_score = classify_setup(
                self.settings,
                primary_catalyst["catalyst_direction"],
                crowding_record,
                premarket,
                squeeze_risk,
            )
            as_of_timestamp = datetime.combine(trade_date, time(hour=13, minute=15), tzinfo=timezone.utc).isoformat()
            watchlist_rows.append(
                {
                    "date": trade_date.isoformat(),
                    "ticker": ticker,
                    "catalyst_bucket": primary_catalyst["catalyst_bucket"],
                    "catalyst_direction": primary_catalyst["catalyst_direction"],
                    "catalyst_detail": primary_catalyst["catalyst_detail"],
                    "pre_market_gap_pct": premarket.gap_pct,
                    "pre_market_relative_volume": premarket.relative_volume,
                    "pre_market_dollar_volume": premarket.dollar_volume,
                    "attention": crowding_record["attention"],
                    "positioning_pressure": crowding_record["positioning_pressure"],
                    "crowd_trap_risk": crowding_record["crowd_trap_risk"],
                    "short_interest_pct_float": None if latest_short_interest is None else latest_short_interest["short_pct_float"],
                    "days_to_cover": None if latest_short_interest is None else latest_short_interest["days_to_cover"],
                    "suggested_setup": suggested_setup,
                    "setup_score": setup_score,
                    "short_tradable": int(squeeze_risk.tradable),
                    "as_of_timestamp": as_of_timestamp,
                }
            )

        self.db.upsert_crowding_index(crowding_rows)
        self.db.upsert_watchlist(watchlist_rows)
        return len(watchlist_rows)


def run_scanner_from_config(trade_date: date, settings_path: Path = Path("config/settings.yaml")) -> int:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    scanner = HistoricalScanner(settings=settings, db=db)
    return scanner.scan_day(trade_date)
