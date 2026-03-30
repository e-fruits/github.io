from datetime import date
from pathlib import Path

from src.signals.scanner import HistoricalScanner
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_scanner_generates_watchlist_row(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "scanner.sqlite"
    settings.database.path = db_path
    db = DatabaseManager(db_path)
    db.initialize()

    db.upsert_universe(
        [
            {
                "date": "2024-01-22",
                "ticker": "ABCD",
                "company_name": "Alpha Beta Corp",
                "market_cap": 750_000_000,
                "sector": "Technology",
                "exchange": "NASDAQ",
                "avg_volume_20d": 1_000_000,
                "avg_dollar_volume_20d": 10_000_000,
                "is_active": 1,
                "as_of_timestamp": "2024-01-22T13:00:00+00:00",
            }
        ]
    )
    db.upsert_daily_prices(
        [
            {
                "date": "2024-01-10",
                "ticker": "ABCD",
                "open": 45.0,
                "high": 46.0,
                "low": 44.5,
                "close": 45.5,
                "volume": 900000,
                "vwap": 45.4,
                "dollar_volume": 40950000,
                "relative_volume_20d": 1.1,
                "gap_pct": 0.02,
                "intraday_range_pct": 0.033,
                "pre_market_high": 45.7,
                "pre_market_low": 45.1,
                "pre_market_volume": 70000,
                "pre_market_dollar_volume": 3200000,
                "pre_market_relative_volume": 1.2,
                "as_of_timestamp": "2024-01-10T20:00:00+00:00",
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
    db.upsert_catalysts(
        [
            {
                "date": "2024-01-22",
                "ticker": "ABCD",
                "catalyst_bucket": "earnings",
                "catalyst_detail": "Quarterly earnings report",
                "catalyst_direction": "positive",
                "pre_market_gap_pct": 0.06,
                "source": "finnhub",
                "source_timestamp": "2024-01-22T12:00:00+00:00",
                "as_of_timestamp": "2024-01-22T12:05:00+00:00",
            }
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

    scanner = HistoricalScanner(settings=settings, db=db)
    count = scanner.scan_day(date(2024, 1, 22))

    assert count == 1
    with db.connect() as connection:
        row = connection.execute("SELECT suggested_setup, attention, positioning_pressure FROM watchlist WHERE ticker='ABCD'").fetchone()
    assert row is not None
    assert row["suggested_setup"] in {"long_continuation", "long_mean_reversion", "skip", "short_continuation", "short_mean_reversion"}
