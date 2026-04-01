"""Point-in-time universe construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

from src.utils.config import AppSettings, load_settings
from src.utils.db import DatabaseManager
from src.utils.http import HttpRequestError, JsonHttpClient
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


class PolygonReferenceClient:
    """Massive-backed historical ticker universe client."""

    def __init__(self, settings: AppSettings) -> None:
        api_key = settings.providers.polygon.api_key
        if not api_key:
            raise ValueError("Massive API key is required for the live universe client")
        self.settings = settings
        self.http = JsonHttpClient(
            base_url=settings.providers.polygon.base_url,
            cache_dir=settings.pipeline.cache_dir,
            timeout_seconds=settings.pipeline.request_timeout_seconds,
            auth_query_param="apiKey",
            auth_token=api_key,
        )

    def get_tickers_for_date(self, as_of_date: date) -> Iterable[UniverseCandidate]:
        ticker_rows = self.http.get_paginated_results(
            "/v3/reference/tickers",
            params={
                "market": "stocks",
                "active": "true",
                "date": as_of_date.isoformat(),
                "limit": 1000,
                "sort": "ticker",
                "order": "asc",
            },
            cache_namespace=f"polygon/reference_tickers/{as_of_date.isoformat()}",
        )

        for ticker_row in ticker_rows:
            ticker = str(ticker_row.get("ticker", "")).upper()
            if not ticker:
                continue

            details = self._fetch_details(ticker, as_of_date)
            if details is None:
                continue

            metrics = self._fetch_liquidity_metrics(ticker, as_of_date)
            if metrics is None:
                continue

            exchange = self._normalize_exchange(
                details.get("primary_exchange") or ticker_row.get("primary_exchange") or ticker_row.get("exchange")
            )
            if exchange is None:
                continue

            market_cap = details.get("market_cap")
            if market_cap is None:
                weighted_shares = details.get("weighted_shares_outstanding")
                price = metrics["last_close"]
                market_cap = None if weighted_shares in (None, 0) or price is None else float(weighted_shares) * float(price)
            if market_cap is None:
                continue

            timestamp_value = details.get("updated_utc") or ticker_row.get("last_updated_utc")
            as_of_timestamp = self._parse_timestamp(timestamp_value, fallback_date=as_of_date)

            yield UniverseCandidate(
                date=as_of_date,
                ticker=ticker,
                company_name=str(details.get("name") or ticker_row.get("name") or ticker),
                market_cap=float(market_cap),
                sector=self._sector_from_details(details),
                exchange=exchange,
                avg_volume_20d=metrics["avg_volume_20d"],
                avg_dollar_volume_20d=metrics["avg_dollar_volume_20d"],
                is_active=bool(details.get("active", ticker_row.get("active", True))),
                price=metrics["last_close"],
                as_of_timestamp=as_of_timestamp,
            )

    def _fetch_details(self, ticker: str, as_of_date: date) -> dict[str, object] | None:
        try:
            payload = self.http.get_json(
                f"/v3/reference/tickers/{ticker}",
                params={"date": as_of_date.isoformat()},
                cache_namespace=f"polygon/ticker_details/{ticker}/{as_of_date.isoformat()}",
            )
        except HttpRequestError as exc:
            LOGGER.warning("Massive ticker details failed for %s on %s: %s", ticker, as_of_date, exc)
            return None
        result = payload.get("results")
        return result if isinstance(result, dict) else None

    def _fetch_liquidity_metrics(self, ticker: str, as_of_date: date) -> dict[str, float] | None:
        start_date = as_of_date - timedelta(days=max(self.settings.price_volume.rolling_window_days * 3, 40))
        try:
            payload = self.http.get_json(
                f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date.isoformat()}/{as_of_date.isoformat()}",
                params={"adjusted": str(self.settings.price_volume.adjusted).lower(), "sort": "asc", "limit": 5000},
                cache_namespace=f"polygon/daily_aggs/{ticker}",
            )
        except HttpRequestError as exc:
            LOGGER.warning("Massive daily aggs failed for %s: %s", ticker, exc)
            return None

        rows = payload.get("results", [])
        if not isinstance(rows, list) or len(rows) < self.settings.price_volume.rolling_window_days:
            return None

        parsed_rows = [row for row in rows if isinstance(row, dict)]
        if not parsed_rows:
            return None

        trailing_rows = parsed_rows[-self.settings.price_volume.rolling_window_days :]
        last_close = trailing_rows[-1].get("c")
        if last_close is None:
            return None

        avg_volume = sum(float(row.get("v", 0.0)) for row in trailing_rows) / len(trailing_rows)
        avg_dollar_volume = (
            sum(float(row.get("vw", row.get("c", 0.0))) * float(row.get("v", 0.0)) for row in trailing_rows) / len(trailing_rows)
        )
        return {
            "avg_volume_20d": avg_volume,
            "avg_dollar_volume_20d": avg_dollar_volume,
            "last_close": float(last_close),
        }

    @staticmethod
    def _normalize_exchange(value: object) -> str | None:
        if value is None:
            return None
        raw = str(value).upper()
        if raw in {"NASDAQ", "XNAS", "NAS", "Q"}:
            return "NASDAQ"
        if raw in {"NYSE", "XNYS", "N"}:
            return "NYSE"
        return raw

    @staticmethod
    def _sector_from_details(details: dict[str, object]) -> str | None:
        for key in ("sic_description", "market", "locale"):
            value = details.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _parse_timestamp(value: object, fallback_date: date) -> datetime:
        if isinstance(value, str) and value:
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.combine(fallback_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=8)


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
    reference_client: ReferenceTickerClient
    if settings.providers.polygon.api_key:
        reference_client = PolygonReferenceClient(settings)
    else:
        if not settings.pipeline.use_stub_fallback:
            raise ValueError("Massive API key is required when stub fallback is disabled")
        LOGGER.warning("Massive API key not configured; falling back to stub universe client")
        reference_client = StubPolygonReferenceClient()
    builder = UniverseBuilder(
        settings=settings,
        db=db,
        reference_client=reference_client,
    )
    return builder.build_range(start_date, end_date)
