from datetime import date
from pathlib import Path

from src.pipeline.data_audit import DataAuditService
from src.utils.config import load_settings
from src.utils.db import DatabaseManager


def test_data_audit_flags_lookahead_and_missing_fields(tmp_path: Path) -> None:
    settings = load_settings(Path("config/settings.yaml"))
    db_path = tmp_path / "audit.sqlite"
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
                "date": "2024-01-22",
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
                "as_of_timestamp": "2024-01-22T20:00:00+00:00",
            }
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
                "pre_market_gap_pct": 0.08,
                "source": "finnhub",
                "source_timestamp": "2024-01-22T12:00:00+00:00",
                "as_of_timestamp": "2024-01-22T12:05:00+00:00",
            }
        ]
    )

    service = DataAuditService(db)
    report = service.audit("ABCD", date(2024, 1, 22))

    assert any(item.dataset == "universe" for item in report.available_fields)
    assert any(item.dataset == "catalysts[1]" for item in report.available_fields)
    assert any(item.dataset == "daily_prices" for item in report.lookahead_violations)
    assert any(item.dataset.startswith("wsb_mentions") or item.dataset == "wsb_mentions" for item in report.missing_fields)
