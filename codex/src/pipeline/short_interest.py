"""FINRA short interest ingestion with explicit publication-lag handling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

from src.utils.config import AppSettings, load_settings
from src.utils.db import DatabaseManager
from src.utils.timestamps import ensure_utc_timestamp

LOGGER = logging.getLogger(__name__)


@dataclass
class ShortInterestReport:
    report_date: date
    settlement_date: date
    ticker: str
    short_interest_shares: float
    shares_outstanding: float
    float_shares: float
    avg_daily_volume_20d: float
    as_of_timestamp: datetime


class ShortInterestClient(Protocol):
    def get_reports(self, ticker: str, start_date: date, end_date: date) -> Iterable[ShortInterestReport]:
        """Return short-interest reports published in the requested period."""


class StubShortInterestClient:
    def __init__(self, publication_lag_business_days: int = 8) -> None:
        self.publication_lag_business_days = publication_lag_business_days

    def get_reports(self, ticker: str, start_date: date, end_date: date) -> Iterable[ShortInterestReport]:
        current = date(start_date.year, start_date.month, 1)
        while current <= end_date:
            settlement_date = current.replace(day=15) if current.day == 1 else current
            report_date = self._add_business_days(settlement_date, self.publication_lag_business_days)
            if start_date <= report_date <= end_date:
                base = 1_200_000 + (sum(ord(char) for char in ticker) % 100_000)
                yield ShortInterestReport(
                    report_date=report_date,
                    settlement_date=settlement_date,
                    ticker=ticker,
                    short_interest_shares=float(base),
                    shares_outstanding=25_000_000.0,
                    float_shares=18_000_000.0,
                    avg_daily_volume_20d=950_000.0,
                    as_of_timestamp=datetime.combine(report_date, time(hour=21, minute=0), tzinfo=timezone.utc),
                )
            next_month = current.month + 1
            next_year = current.year + (1 if next_month == 13 else 0)
            current = date(next_year, 1 if next_month == 13 else next_month, 1)

    @staticmethod
    def _add_business_days(start: date, days: int) -> date:
        current = start
        added = 0
        while added < days:
            current += timedelta(days=1)
            if current.weekday() < 5:
                added += 1
        return current


class ShortInterestLoader:
    def __init__(self, settings: AppSettings, db: DatabaseManager, client: ShortInterestClient) -> None:
        self.settings = settings
        self.db = db
        self.client = client

    def load_range(self, start_date: date, end_date: date, tickers: Iterable[str] | None = None) -> int:
        if tickers is None:
            tickers = self.db.fetch_universe_tickers(start_date.isoformat(), end_date.isoformat())

        rows: list[dict[str, object]] = []
        for ticker in tickers:
            for report in self.client.get_reports(ticker, start_date, end_date):
                ensure_utc_timestamp(report.as_of_timestamp, label=f"{ticker}.{report.report_date}.short_interest_as_of")
                short_pct_float = None if report.float_shares == 0 else report.short_interest_shares / report.float_shares
                days_to_cover = None if report.avg_daily_volume_20d == 0 else report.short_interest_shares / report.avg_daily_volume_20d
                rows.append(
                    {
                        "report_date": report.report_date.isoformat(),
                        "settlement_date": report.settlement_date.isoformat(),
                        "ticker": report.ticker,
                        "short_interest_shares": report.short_interest_shares,
                        "shares_outstanding": report.shares_outstanding,
                        "float_shares": report.float_shares,
                        "short_pct_float": short_pct_float,
                        "days_to_cover": days_to_cover,
                        "as_of_timestamp": report.as_of_timestamp.isoformat(),
                    }
                )

        self.db.upsert_short_interest(rows)
        LOGGER.info("Stored short-interest rows=%s", len(rows))
        return len(rows)

    def latest_available_as_of(self, ticker: str, trade_date: date):
        as_of_timestamp = datetime.combine(trade_date, time(hour=23, minute=59), tzinfo=timezone.utc)
        return self.db.fetch_latest_short_interest_as_of(ticker, as_of_timestamp)


def load_short_interest_from_config(
    start_date: date,
    end_date: date,
    settings_path: Path = Path("config/settings.yaml"),
    tickers: Iterable[str] | None = None,
) -> int:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    loader = ShortInterestLoader(
        settings=settings,
        db=db,
        client=StubShortInterestClient(settings.short_interest.report_publication_lag_business_days),
    )
    return loader.load_range(start_date=start_date, end_date=end_date, tickers=tickers)
