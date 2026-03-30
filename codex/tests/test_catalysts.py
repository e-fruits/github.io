from datetime import date
from pathlib import Path

from src.pipeline.catalysts import CatalystLoader, StubCatalystClient
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_catalyst_loader_stores_source_timestamp_and_gap(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "catalysts.sqlite"
    settings.database.path = db_path
    db = DatabaseManager(db_path)
    db.initialize()

    db.upsert_universe(
        [
            {
                "date": "2024-01-01",
                "ticker": "ABCD",
                "company_name": "Alpha Beta Corp",
                "market_cap": 750_000_000,
                "sector": "Technology",
                "exchange": "NASDAQ",
                "avg_volume_20d": 1_000_000,
                "avg_dollar_volume_20d": 10_000_000,
                "is_active": 1,
                "as_of_timestamp": "2024-01-01T13:00:00+00:00",
            }
        ]
    )

    db.upsert_daily_prices(
        [
            {
                "date": "2024-01-01",
                "ticker": "ABCD",
                "open": 11.0,
                "high": 12.0,
                "low": 10.5,
                "close": 11.8,
                "volume": 1_500_000,
                "vwap": 11.4,
                "dollar_volume": 17_700_000,
                "relative_volume_20d": 1.5,
                "gap_pct": 0.08,
                "intraday_range_pct": 0.136,
                "pre_market_high": 11.3,
                "pre_market_low": 10.9,
                "pre_market_volume": 80_000,
                "pre_market_dollar_volume": 900_000,
                "pre_market_relative_volume": 0.08,
                "as_of_timestamp": "2024-01-01T20:00:00+00:00",
            }
        ]
    )

    loader = CatalystLoader(settings=settings, db=db, client=StubCatalystClient())
    inserted = loader.load_ticker("ABCD", date(2024, 1, 1), date(2024, 1, 31))

    assert inserted > 0

    with db.connect() as connection:
        row = connection.execute(
            """
            SELECT catalyst_bucket, catalyst_direction, pre_market_gap_pct, source_timestamp
            FROM catalysts
            WHERE ticker = 'ABCD'
            ORDER BY date ASC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row["catalyst_bucket"] in {"earnings", "analyst", "regulatory", "news"}
    assert row["catalyst_direction"] in {"positive", "negative", "ambiguous"}
    assert row["pre_market_gap_pct"] == 0.08
    assert "T" in row["source_timestamp"]
