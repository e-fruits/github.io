from pathlib import Path

from src.pipeline.universe import UniverseBuilder, StubPolygonReferenceClient
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_universe_builder_applies_blocklist(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "test.sqlite"
    settings.database.path = db_path
    db = DatabaseManager(db_path)
    db.initialize()

    builder = UniverseBuilder(settings=settings, db=db, reference_client=StubPolygonReferenceClient())
    inserted = builder.build_for_date(__import__("datetime").date(2024, 1, 2))

    assert inserted == 1

