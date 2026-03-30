from datetime import date
from pathlib import Path

from src.pipeline.short_interest import ShortInterestLoader, StubShortInterestClient
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_short_interest_loader_uses_latest_published_report(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "short_interest.sqlite"
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

    loader = ShortInterestLoader(settings=settings, db=db, client=StubShortInterestClient(8))
    inserted = loader.load_range(date(2024, 1, 1), date(2024, 3, 31), tickers=["ABCD"])

    assert inserted > 0

    before_publication = loader.latest_available_as_of("ABCD", date(2024, 1, 17))
    after_publication = loader.latest_available_as_of("ABCD", date(2024, 1, 29))

    assert before_publication is None
    assert after_publication is not None
    assert after_publication["short_pct_float"] is not None
    assert after_publication["days_to_cover"] is not None
