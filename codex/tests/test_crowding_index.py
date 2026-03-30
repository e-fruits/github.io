from datetime import date
from pathlib import Path

from src.signals.crowding_index import build_crowding_index_record
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_crowding_index_builds_scores_and_labels(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "crowding.sqlite"
    settings.database.path = db_path
    db = DatabaseManager(db_path)
    db.initialize()

    db.upsert_daily_prices(
        [
            {
                "date": "2024-01-20",
                "ticker": "ABCD",
                "open": 48.5,
                "high": 49.2,
                "low": 47.8,
                "close": 48.9,
                "volume": 1000000,
                "vwap": 48.7,
                "dollar_volume": 48900000,
                "relative_volume_20d": 1.2,
                "gap_pct": 0.03,
                "intraday_range_pct": 0.028,
                "pre_market_high": 49.8,
                "pre_market_low": 48.7,
                "pre_market_volume": 90000,
                "pre_market_dollar_volume": 4400000,
                "pre_market_relative_volume": 1.5,
                "as_of_timestamp": "2024-01-20T20:00:00+00:00",
            },
            {
                "date": "2024-01-22",
                "ticker": "ABCD",
                "open": 49.0,
                "high": 50.2,
                "low": 48.9,
                "close": 49.8,
                "volume": 1500000,
                "vwap": 49.5,
                "dollar_volume": 74700000,
                "relative_volume_20d": 1.6,
                "gap_pct": 0.06,
                "intraday_range_pct": 0.026,
                "pre_market_high": 50.0,
                "pre_market_low": 49.2,
                "pre_market_volume": 110000,
                "pre_market_dollar_volume": 5400000,
                "pre_market_relative_volume": 1.8,
                "as_of_timestamp": "2024-01-22T20:00:00+00:00",
            },
        ]
    )
    db.upsert_wsb_mentions(
        [
            {
                "date": "2024-01-22",
                "ticker": "ABCD",
                "mention_count": 50,
                "mention_count_prior_day": 20,
                "mention_velocity_pct": 1.5,
                "mention_vs_baseline": 2.2,
                "sentiment_score": 0.6,
                "sentiment_unanimity": 0.8,
                "upvotes": 2000,
                "rank": 1,
                "rank_change_24h": -1,
                "as_of_timestamp": "2024-01-22T13:00:00+00:00",
                "source": "apewisdom",
            }
        ]
    )
    db.upsert_options_flow_daily(
        [
            {
                "date": "2024-01-22",
                "ticker": "ABCD",
                "total_call_volume": 400,
                "total_put_volume": 100,
                "call_put_ratio": 4.0,
                "total_call_oi": 800,
                "total_put_oi": 300,
                "oi_change_calls": 50,
                "oi_change_puts": 10,
                "unusual_volume_ratio": 2.5,
                "small_lot_call_volume": 200,
                "small_lot_put_volume": 40,
                "small_lot_call_pct": 0.5,
                "as_of_timestamp": "2024-01-22T21:00:00+00:00",
            }
        ]
    )
    db.upsert_short_interest(
        [
            {
                "report_date": "2024-01-19",
                "settlement_date": "2024-01-10",
                "ticker": "ABCD",
                "short_interest_shares": 3000000,
                "shares_outstanding": 25000000,
                "float_shares": 18000000,
                "short_pct_float": 0.166,
                "days_to_cover": 4.0,
                "as_of_timestamp": "2024-01-19T21:00:00+00:00",
            }
        ]
    )

    record = build_crowding_index_record(settings, db, "ABCD", date(2024, 1, 22))

    assert record["attention"] in {"low", "medium", "high"}
    assert record["positioning_pressure"] in {"low", "medium", "high"}
    assert record["crowd_trap_risk"] in {"absent", "present"}
    assert 0.0 <= record["attention_score"] <= 1.0
    assert 0.0 <= record["positioning_score"] <= 1.0
    assert 0.0 <= record["trap_score"] <= 1.0
