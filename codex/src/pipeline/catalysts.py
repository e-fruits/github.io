"""Historical catalyst ingestion with explicit point-in-time timestamps."""

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
class CatalystEvent:
    trade_date: date
    ticker: str
    event_type: str
    detail: str
    source: str
    source_timestamp: datetime
    as_of_timestamp: datetime
    direction_hint: str | None = None
    analyst_firm: str | None = None
    price_target: float | None = None
    actual_value: float | None = None
    estimate_value: float | None = None


class CatalystClient(Protocol):
    def get_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        """Return historical catalyst events for a ticker."""


class StubCatalystClient:
    """Synthetic client that mimics Finnhub / Yahoo-style catalyst records."""

    def get_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        current = start_date
        while current <= end_date:
            if current.weekday() < 5 and current.day in {1, 8, 15, 22}:
                base_time = datetime.combine(current, time(hour=12, minute=30), tzinfo=timezone.utc)
                if current.day % 22 == 0:
                    yield CatalystEvent(
                        trade_date=current,
                        ticker=ticker,
                        event_type="analyst_upgrade",
                        detail="Upgrade by North Street to Buy",
                        source="finnhub",
                        source_timestamp=base_time - timedelta(minutes=5),
                        as_of_timestamp=base_time,
                        direction_hint="positive",
                        analyst_firm="North Street",
                        price_target=18.0,
                    )
                elif current.day % 15 == 0:
                    yield CatalystEvent(
                        trade_date=current,
                        ticker=ticker,
                        event_type="fda_decision",
                        detail="FDA accepts supplemental filing",
                        source="finnhub",
                        source_timestamp=base_time - timedelta(minutes=20),
                        as_of_timestamp=base_time,
                        direction_hint="positive",
                    )
                elif current.day % 8 == 0:
                    yield CatalystEvent(
                        trade_date=current,
                        ticker=ticker,
                        event_type="company_news",
                        detail="Announces material commercial partnership",
                        source="yahoo_finance",
                        source_timestamp=base_time - timedelta(minutes=10),
                        as_of_timestamp=base_time,
                        direction_hint="positive",
                    )
                else:
                    yield CatalystEvent(
                        trade_date=current,
                        ticker=ticker,
                        event_type="earnings",
                        detail="Quarterly earnings report",
                        source="finnhub",
                        source_timestamp=base_time - timedelta(minutes=30),
                        as_of_timestamp=base_time,
                        direction_hint="positive",
                        actual_value=0.42,
                        estimate_value=0.31,
                    )
            current += timedelta(days=1)


class CatalystLoader:
    def __init__(self, settings: AppSettings, db: DatabaseManager, client: CatalystClient) -> None:
        self.settings = settings
        self.db = db
        self.client = client

    def load_range(self, start_date: date, end_date: date, tickers: Iterable[str] | None = None) -> int:
        if tickers is None:
            tickers = self.db.fetch_universe_tickers(start_date.isoformat(), end_date.isoformat())

        total_rows = 0
        for ticker in tickers:
            total_rows += self.load_ticker(ticker, start_date, end_date)

        LOGGER.info("Stored catalyst rows=%s from %s to %s", total_rows, start_date, end_date)
        return total_rows

    def load_ticker(self, ticker: str, start_date: date, end_date: date) -> int:
        rows: list[dict[str, object]] = []
        for event in self.client.get_events(ticker, start_date, end_date):
            if not self._is_enabled(event.event_type):
                continue
            ensure_utc_timestamp(event.source_timestamp, label=f"{ticker}.{event.trade_date}.source_timestamp")
            ensure_utc_timestamp(event.as_of_timestamp, label=f"{ticker}.{event.trade_date}.as_of_timestamp")

            rows.append(
                {
                    "date": event.trade_date.isoformat(),
                    "ticker": event.ticker,
                    "catalyst_bucket": self._bucket_for_event(event.event_type),
                    "catalyst_detail": self._build_detail(event),
                    "catalyst_direction": self._direction_for_event(event),
                    "pre_market_gap_pct": self.db.fetch_daily_gap_pct(ticker, event.trade_date.isoformat()),
                    "source": event.source,
                    "source_timestamp": event.source_timestamp.isoformat(),
                    "as_of_timestamp": event.as_of_timestamp.isoformat(),
                }
            )

        self.db.upsert_catalysts(rows)
        LOGGER.info("Stored %s catalyst rows for %s", len(rows), ticker)
        return len(rows)

    def _is_enabled(self, event_type: str) -> bool:
        flags = self.settings.catalysts
        if event_type == "earnings":
            return flags.include_earnings
        if event_type.startswith("analyst_"):
            return flags.include_analyst_actions
        if event_type.startswith("fda_"):
            return flags.include_fda_events
        return flags.include_company_news

    @staticmethod
    def _bucket_for_event(event_type: str) -> str:
        if event_type == "earnings":
            return "earnings"
        if event_type.startswith("analyst_"):
            return "analyst"
        if event_type.startswith("fda_"):
            return "regulatory"
        return "news"

    @staticmethod
    def _direction_for_event(event: CatalystEvent) -> str:
        if event.direction_hint in {"positive", "negative", "ambiguous"}:
            return event.direction_hint
        if event.event_type == "earnings" and event.actual_value is not None and event.estimate_value is not None:
            if event.actual_value > event.estimate_value:
                return "positive"
            if event.actual_value < event.estimate_value:
                return "negative"
        if "upgrade" in event.event_type:
            return "positive"
        if "downgrade" in event.event_type:
            return "negative"
        return "ambiguous"

    @staticmethod
    def _build_detail(event: CatalystEvent) -> str:
        parts = [event.detail]
        if event.event_type == "earnings" and event.actual_value is not None and event.estimate_value is not None:
            parts.append(f"actual={event.actual_value}")
            parts.append(f"estimate={event.estimate_value}")
        if event.analyst_firm:
            parts.append(f"firm={event.analyst_firm}")
        if event.price_target is not None:
            parts.append(f"target={event.price_target}")
        return " | ".join(parts)


def load_catalysts_from_config(
    start_date: date,
    end_date: date,
    settings_path: Path = Path("config/settings.yaml"),
    tickers: Iterable[str] | None = None,
) -> int:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    loader = CatalystLoader(settings=settings, db=db, client=StubCatalystClient())
    return loader.load_range(start_date=start_date, end_date=end_date, tickers=tickers)
