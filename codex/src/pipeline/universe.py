"""Point-in-time universe construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

from src.utils.config import AppSettings, load_settings
from src.utils.db import DatabaseManager
from src.utils.timestamps import ensure_utc_timestamp, trading_days

LOGGER = logging.getLogger(__name__)


@dataclass
class UniverseCandidate:
    date: date
    ticker: str
    company_name: str
    market_cap: float
    sector: str | None
    exchange: str
    avg_volume_20d: float
    avg_dollar_volume_20d: float
    is_active: bool
    price: float
    as_of_timestamp: datetime


class ReferenceTickerClient(Protocol):
    def get_tickers_for_date(self, as_of_date: date) -> Iterable[UniverseCandidate]:
        """Return all historically available tickers for the requested date."""


class StubPolygonReferenceClient:
    """Local stub to keep the builder runnable before wiring the real API."""

    def get_tickers_for_date(self, as_of_date: date) -> Iterable[UniverseCandidate]:
        base_timestamp = datetime.combine(as_of_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=8)
        return [
            UniverseCandidate(
                date=as_of_date,
                ticker="ABCD",
                company_name="Alpha Beta Corp",
                market_cap=750_000_000,
                sector="Technology",
                exchange="NASDAQ",
                avg_volume_20d=1_500_000,
                avg_dollar_volume_20d=18_000_000,
                is_active=True,
                price=12.4,
                as_of_timestamp=base_timestamp,
            ),
            UniverseCandidate(
                date=as_of_date,
                ticker="MOCK1",
                company_name="Blocked Name Inc",
                market_cap=900_000_000,
                sector="Technology",
                exchange="NASDAQ",
                avg_volume_20d=2_000_000,
                avg_dollar_volume_20d=22_000_000,
                is_active=True,
                price=8.2,
                as_of_timestamp=base_timestamp,
            ),
            UniverseCandidate(
                date=as_of_date,
                ticker="TINY",
                company_name="Too Small Corp",
                market_cap=120_000_000,
                sector="Healthcare",
                exchange="NYSE",
                avg_volume_20d=900_000,
                avg_dollar_volume_20d=6_300_000,
                is_active=True,
                price=7.0,
                as_of_timestamp=base_timestamp,
            ),
        ]


class UniverseBuilder:
    def __init__(
        self,
        settings: AppSettings,
        db: DatabaseManager,
        reference_client: ReferenceTickerClient,
    ) -> None:
        self.settings = settings
        self.db = db
        self.reference_client = reference_client

    def build_range(self, start_date: date, end_date: date) -> int:
        total_inserted = 0
        for trade_date in trading_days(start_date, end_date):
            total_inserted += self.build_for_date(trade_date)
        LOGGER.info("Built universe rows=%s from %s to %s", total_inserted, start_date, end_date)
        return total_inserted

    def build_for_date(self, trade_date: date) -> int:
        raw_candidates = list(self.reference_client.get_tickers_for_date(trade_date))
        LOGGER.info("Fetched raw universe candidates=%s for %s", len(raw_candidates), trade_date)

        filtered = [candidate for candidate in raw_candidates if self._passes_filters(candidate)]
        rows = [self._to_record(candidate) for candidate in filtered]
        self.db.upsert_universe(rows)
        LOGGER.info("Stored filtered universe rows=%s for %s", len(rows), trade_date)
        return len(rows)

    def _passes_filters(self, candidate: UniverseCandidate) -> bool:
        universe_cfg = self.settings.universe
        blocklist = self.settings.compliance_blocklist

        ensure_utc_timestamp(candidate.as_of_timestamp, label=f"{candidate.ticker}.as_of_timestamp")

        if candidate.ticker in blocklist.blocked_tickers:
            LOGGER.info("Excluded %s due to ticker blocklist", candidate.ticker)
            return False
        if candidate.sector and candidate.sector in blocklist.blocked_sectors:
            LOGGER.info("Excluded %s due to sector blocklist=%s", candidate.ticker, candidate.sector)
            return False
        if candidate.exchange not in universe_cfg.allowed_exchanges:
            return False
        if not universe_cfg.include_delisted and not candidate.is_active:
            return False
        if not (universe_cfg.min_market_cap <= candidate.market_cap <= universe_cfg.max_market_cap):
            return False
        if candidate.price <= universe_cfg.min_price:
            return False
        if candidate.avg_volume_20d < universe_cfg.min_avg_volume_20d:
            return False
        return True

    @staticmethod
    def _to_record(candidate: UniverseCandidate) -> dict[str, object]:
        return {
            "date": candidate.date.isoformat(),
            "ticker": candidate.ticker,
            "company_name": candidate.company_name,
            "market_cap": candidate.market_cap,
            "sector": candidate.sector,
            "exchange": candidate.exchange,
            "avg_volume_20d": candidate.avg_volume_20d,
            "avg_dollar_volume_20d": candidate.avg_dollar_volume_20d,
            "is_active": int(candidate.is_active),
            "as_of_timestamp": candidate.as_of_timestamp.isoformat(),
        }


def build_universe_from_config(
    start_date: date,
    end_date: date,
    settings_path: Path = Path("config/settings.yaml"),
) -> int:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    builder = UniverseBuilder(
        settings=settings,
        db=db,
        reference_client=StubPolygonReferenceClient(),
    )
    return builder.build_range(start_date, end_date)
