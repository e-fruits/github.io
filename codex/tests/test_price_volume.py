from datetime import date
from pathlib import Path

from src.pipeline.price_volume import PriceVolumeLoader, StubHistoricalPriceClient
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_price_volume_loader_computes_expected_fields(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "prices.sqlite"
    settings.database.path = db_path
    db = DatabaseManager(db_path)
    db.initialize()

    db.upsert_universe(
        [
            {
                "date": "2024-02-01",
                "ticker": "ABCD",
                "company_name": "Alpha Beta Corp",
                "market_cap": 750_000_000,
                "sector": "Technology",
                "exchange": "NASDAQ",
                "avg_volume_20d": 1_000_000,
                "avg_dollar_volume_20d": 10_000_000,
                "is_active": 1,
                "as_of_timestamp": "2024-02-01T13:00:00+00:00",
            }
        ]
    )

    loader = PriceVolumeLoader(settings=settings, db=db, client=StubHistoricalPriceClient())
    inserted = loader.load_ticker("ABCD", date(2024, 2, 1), date(2024, 2, 29))

    assert inserted > 0

    with db.connect() as connection:
        row = connection.execute(
            """
            SELECT relative_volume_20d, gap_pct, intraday_range_pct, pre_market_volume
            FROM daily_prices
            WHERE ticker = 'ABCD'
            ORDER BY date DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row["relative_volume_20d"] is not None
    assert row["gap_pct"] is not None
    assert row["intraday_range_pct"] is not None
    assert row["pre_market_volume"] is not None
