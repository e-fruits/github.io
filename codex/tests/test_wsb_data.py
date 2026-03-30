from datetime import date
from pathlib import Path

from src.pipeline.wsb_data import WsbDataLoader
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_apewisdom_rollups_compute_velocity_and_baseline(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "wsb.sqlite"
    settings.database.path = db_path
    settings.wsb.sources = ["apewisdom"]
    db = DatabaseManager(db_path)
    db.initialize()

    loader = WsbDataLoader(settings=settings, db=db)
    inserted = loader.load_apewisdom_range(date(2024, 1, 1), date(2024, 2, 15))

    assert inserted > 0

    with db.connect() as connection:
        row = connection.execute(
            """
            SELECT mention_count_prior_day, mention_velocity_pct, mention_vs_baseline, upvotes, rank
            FROM wsb_mentions
            WHERE ticker = 'ABCD' AND source = 'apewisdom'
            ORDER BY date DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row["mention_count_prior_day"] is not None
    assert row["mention_velocity_pct"] is not None
    assert row["mention_vs_baseline"] is not None
    assert row["upvotes"] is not None
    assert row["rank"] is not None


def test_praw_rollups_store_sentiment_and_unanimity(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "praw.sqlite"
    settings.database.path = db_path
    settings.wsb.sources = ["praw"]
    db = DatabaseManager(db_path)
    db.initialize()

    loader = WsbDataLoader(settings=settings, db=db)
    inserted = loader.load_praw_range(date(2024, 1, 1), date(2024, 1, 31))

    assert inserted > 0

    with db.connect() as connection:
        row = connection.execute(
            """
            SELECT mention_count, sentiment_score, sentiment_unanimity, upvotes
            FROM wsb_mentions
            WHERE ticker = 'ABCD' AND source = 'praw'
            ORDER BY date DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row["mention_count"] is not None
    assert row["sentiment_score"] is not None
    assert row["sentiment_unanimity"] is not None
    assert row["upvotes"] is not None
