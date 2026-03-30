from datetime import date
from pathlib import Path

from src.pipeline.options_flow import OptionsFlowLoader, StubOptionsFlowClient
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_options_flow_loader_computes_ratios_and_handles_missing(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "options.sqlite"
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
            },
            {
                "date": "2024-01-01",
                "ticker": "NOOPT",
                "company_name": "No Options Corp",
                "market_cap": 900_000_000,
                "sector": "Healthcare",
                "exchange": "NASDAQ",
                "avg_volume_20d": 800_000,
                "avg_dollar_volume_20d": 8_000_000,
                "is_active": 1,
                "as_of_timestamp": "2024-01-01T13:00:00+00:00",
            },
        ]
    )

    loader = OptionsFlowLoader(settings=settings, db=db, client=StubOptionsFlowClient())
    inserted = loader.load_range(date(2024, 1, 1), date(2024, 2, 15), tickers=["ABCD", "NOOPT"])

    assert inserted > 0

    with db.connect() as connection:
        row = connection.execute(
            """
            SELECT call_put_ratio, unusual_volume_ratio, total_call_volume
            FROM options_flow_daily
            WHERE ticker = 'ABCD'
            ORDER BY date DESC
            LIMIT 1
            """
        ).fetchone()
        missing_row = connection.execute(
            """
            SELECT total_call_volume, total_put_volume
            FROM options_flow_daily
            WHERE ticker = 'NOOPT'
            ORDER BY date DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row["call_put_ratio"] is not None
    assert row["unusual_volume_ratio"] is not None
    assert row["total_call_volume"] is not None
    assert missing_row is not None
    assert missing_row["total_call_volume"] == 0.0
    assert missing_row["total_put_volume"] == 0.0

