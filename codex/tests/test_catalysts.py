from datetime import date
from pathlib import Path

from src.pipeline.catalysts import (
    CatalystLoader,
    CombinedNewsCatalystClient,
    GoogleNewsRssCatalystClient,
    StubCatalystClient,
    YahooFinanceNewsCatalystClient,
    load_catalysts_from_config,
)
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_catalyst_loader_stores_source_timestamp_and_gap(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    settings.catalysts.include_earnings = True
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


def test_yahoo_finance_news_client_parses_rss_items(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    settings.pipeline.cache_dir = tmp_path / "raw_cache"
    client = YahooFinanceNewsCatalystClient(settings)

    class FakeResponse:
        text = """
        <rss version="2.0">
          <channel>
            <item>
              <title>ABCD wins major partnership deal</title>
              <description>Strategic launch expands distribution</description>
              <pubDate>Fri, 02 Jan 2026 14:30:00 +0000</pubDate>
              <source>Yahoo</source>
            </item>
            <item>
              <title>ABCD announces dilution and offering</title>
              <description>Secondary priced overnight</description>
              <pubDate>Sat, 03 Jan 2026 09:00:00 +0000</pubDate>
              <source>Benzinga</source>
            </item>
          </channel>
        </rss>
        """

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int):
        del url, timeout
        return FakeResponse()

    client.session.get = fake_get  # type: ignore[method-assign]
    events = list(client.get_events("ABCD", date(2026, 1, 2), date(2026, 1, 3)))

    assert len(events) == 2
    assert events[0].source == "Yahoo"
    assert events[0].event_type == "company_news"
    assert events[0].direction_hint == "positive"
    assert events[1].source == "Benzinga"
    assert events[1].direction_hint == "negative"


def test_settings_default_to_news_only_catalysts() -> None:
    settings = load_settings(Path("config/settings.yaml"))

    assert settings.catalysts.sources == ["google_news", "yahoo_finance"]
    assert settings.catalysts.include_earnings is False
    assert settings.catalysts.include_analyst_actions is False


def test_load_catalysts_from_config_uses_enabled_news_sources_only(tmp_path: Path, monkeypatch) -> None:
    settings_dir = tmp_path / "config"
    settings_dir.mkdir()
    (settings_dir / "blocked_tickers.yaml").write_text("tickers: []\nsectors: []\nindustries: []\n", encoding="utf-8")
    settings_path = settings_dir / "settings.yaml"
    settings_path.write_text(
        """
database:
  path: "data/test.sqlite"

pipeline:
  cache_dir: "data/raw"
  log_level: "INFO"
  start_date: "2020-01-01"
  request_timeout_seconds: 30
  use_stub_fallback: false

catalysts:
  sources:
    - google_news
  include_earnings: false
  include_analyst_actions: false
  include_fda_events: true
  include_company_news: true

universe:
  min_market_cap: 300000000
  max_market_cap: 5000000000
  min_price: 5.0
  min_avg_volume_20d: 500000
  allowed_exchanges:
    - NYSE

providers:
  yahoo_finance:
    enabled: true
  google_news:
    enabled: true

compliance:
  blocklist_path: "config/blocked_tickers.yaml"
  fail_on_blocklist_conflicts: true
""".strip(),
        encoding="utf-8",
    )

    captured = {}

    class FakeDb:
        def __init__(self, path):
            self.path = path

        def initialize(self) -> None:
            return None

    class FakeLoader:
        def __init__(self, settings, db, client) -> None:
            captured["settings"] = settings
            captured["db"] = db
            captured["client"] = client

        def load_range(self, start_date, end_date, tickers=None) -> int:
            captured["tickers"] = tickers
            return 7

    monkeypatch.setattr("src.pipeline.catalysts.DatabaseManager", FakeDb)
    monkeypatch.setattr("src.pipeline.catalysts.CatalystLoader", FakeLoader)

    inserted = load_catalysts_from_config(date(2026, 1, 2), date(2026, 1, 3), settings_path=settings_path, tickers=["BBAI"])

    assert inserted == 7
    assert isinstance(captured["client"], CombinedNewsCatalystClient)
    assert len(captured["client"].clients) == 1
    assert isinstance(captured["client"].clients[0], GoogleNewsRssCatalystClient)
