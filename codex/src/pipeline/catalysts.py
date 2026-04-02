"""Historical catalyst ingestion with explicit point-in-time timestamps."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol
from urllib.parse import quote_plus

import requests

from src.utils.config import AppSettings, load_settings
from src.utils.db import DatabaseManager
from src.utils.http import HttpRequestError, JsonHttpClient
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


class FinnhubCatalystClient:
    """Finnhub-backed catalyst client for earnings, analyst actions, and news."""

    POSITIVE_NEWS_KEYWORDS = {"beat", "approval", "approves", "wins", "partnership", "upgrade", "raises", "launches"}
    NEGATIVE_NEWS_KEYWORDS = {"miss", "downgrade", "cuts", "delay", "halts", "lawsuit", "offering", "dilution"}
    REGULATORY_KEYWORDS = {"fda", "approval", "clearance", "complete response", "clinical", "phase"}

    def __init__(self, settings: AppSettings) -> None:
        api_key = settings.providers.finnhub.api_key
        if not api_key:
            raise ValueError("Finnhub API key is required for the live catalyst client")
        self.http = JsonHttpClient(
            base_url=settings.providers.finnhub.base_url,
            cache_dir=settings.pipeline.cache_dir,
            timeout_seconds=settings.pipeline.request_timeout_seconds,
            auth_query_param="token",
            auth_token=api_key,
        )

    def get_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        yield from self._earnings_events(ticker, start_date, end_date)
        yield from self._analyst_events(ticker, start_date, end_date)
        yield from self._news_events(ticker, start_date, end_date)

    def _earnings_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        for chunk_start, chunk_end in self._chunk_dates(start_date, end_date, window_days=90):
            try:
                payload = self.http.get_json(
                    "/calendar/earnings",
                    params={"from": chunk_start.isoformat(), "to": chunk_end.isoformat(), "symbol": ticker},
                    cache_namespace=f"finnhub/earnings/{ticker}",
                )
            except HttpRequestError as exc:
                LOGGER.warning("Finnhub earnings calendar failed for %s: %s", ticker, exc)
                return

            events = payload.get("earningsCalendar", [])
            if not isinstance(events, list):
                continue
            for row in events:
                if not isinstance(row, dict) or not row.get("date"):
                    continue
                trade_date = date.fromisoformat(str(row["date"]))
                as_of_timestamp = self._event_timestamp(trade_date, row.get("hour"), default_hour=12, default_minute=0)
                yield CatalystEvent(
                    trade_date=trade_date,
                    ticker=ticker,
                    event_type="earnings",
                    detail="Quarterly earnings report",
                    source="finnhub",
                    source_timestamp=as_of_timestamp,
                    as_of_timestamp=as_of_timestamp,
                    direction_hint=None,
                    actual_value=self._to_float(row.get("epsActual")),
                    estimate_value=self._to_float(row.get("epsEstimate")),
                )

    def _analyst_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        for chunk_start, chunk_end in self._chunk_dates(start_date, end_date, window_days=180):
            try:
                rows = self.http.get_json(
                    "/stock/upgrade-downgrade",
                    params={"symbol": ticker, "from": chunk_start.isoformat(), "to": chunk_end.isoformat()},
                    cache_namespace=f"finnhub/analyst/{ticker}",
                )
            except HttpRequestError as exc:
                LOGGER.warning("Finnhub analyst actions failed for %s: %s", ticker, exc)
                return

            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                grade_time = self._parse_timestamp(row.get("gradeTime"))
                if grade_time is None:
                    continue
                action = str(row.get("action") or "").lower()
                to_grade = str(row.get("toGrade") or row.get("to_grade") or "").strip()
                from_grade = str(row.get("fromGrade") or row.get("from_grade") or "").strip()
                detail = " ".join(part for part in [action.title() if action else "Analyst action", f"{from_grade}->{to_grade}" if from_grade or to_grade else ""] if part).strip()
                yield CatalystEvent(
                    trade_date=grade_time.date(),
                    ticker=ticker,
                    event_type="analyst_upgrade" if "up" in action else "analyst_downgrade" if "down" in action else "analyst_action",
                    detail=detail or "Analyst action",
                    source="finnhub",
                    source_timestamp=grade_time,
                    as_of_timestamp=grade_time,
                    direction_hint="positive" if "up" in action else "negative" if "down" in action else None,
                    analyst_firm=str(row.get("company") or row.get("firm") or "") or None,
                )

    def _news_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        for chunk_start, chunk_end in self._chunk_dates(start_date, end_date, window_days=30):
            try:
                rows = self.http.get_json(
                    "/company-news",
                    params={"symbol": ticker, "from": chunk_start.isoformat(), "to": chunk_end.isoformat()},
                    cache_namespace=f"finnhub/company_news/{ticker}",
                )
            except HttpRequestError as exc:
                LOGGER.warning("Finnhub company news failed for %s: %s", ticker, exc)
                return

            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                timestamp = self._parse_timestamp(row.get("datetime"))
                if timestamp is None:
                    continue
                headline = str(row.get("headline") or "").strip()
                summary = str(row.get("summary") or "").strip()
                combined_text = f"{headline} {summary}".lower()
                is_regulatory = any(keyword in combined_text for keyword in self.REGULATORY_KEYWORDS)
                yield CatalystEvent(
                    trade_date=timestamp.date(),
                    ticker=ticker,
                    event_type="fda_decision" if is_regulatory else "company_news",
                    detail=headline or "Company news",
                    source=str(row.get("source") or "finnhub"),
                    source_timestamp=timestamp,
                    as_of_timestamp=timestamp,
                    direction_hint=self._infer_news_direction(combined_text),
                )

    def _infer_news_direction(self, text: str) -> str:
        positive_hits = sum(1 for keyword in self.POSITIVE_NEWS_KEYWORDS if keyword in text)
        negative_hits = sum(1 for keyword in self.NEGATIVE_NEWS_KEYWORDS if keyword in text)
        if positive_hits > negative_hits:
            return "positive"
        if negative_hits > positive_hits:
            return "negative"
        return "ambiguous"

    @staticmethod
    def _chunk_dates(start_date: date, end_date: date, *, window_days: int) -> Iterable[tuple[date, date]]:
        current = start_date
        while current <= end_date:
            chunk_end = min(current + timedelta(days=window_days - 1), end_date)
            yield current, chunk_end
            current = chunk_end + timedelta(days=1)

    @staticmethod
    def _event_timestamp(trade_date: date, hour_hint: object, *, default_hour: int, default_minute: int) -> datetime:
        hint = str(hour_hint or "").lower()
        if hint == "bmo":
            local_time = time(hour=8, minute=0)
        elif hint == "amc":
            local_time = time(hour=16, minute=5)
        else:
            local_time = time(hour=default_hour, minute=default_minute)
        return datetime.combine(trade_date, local_time, tzinfo=timezone.utc)

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class YahooFinanceNewsCatalystClient:
    """Direct Yahoo Finance RSS-backed news catalyst client."""

    POSITIVE_NEWS_KEYWORDS = FinnhubCatalystClient.POSITIVE_NEWS_KEYWORDS
    NEGATIVE_NEWS_KEYWORDS = FinnhubCatalystClient.NEGATIVE_NEWS_KEYWORDS
    REGULATORY_KEYWORDS = FinnhubCatalystClient.REGULATORY_KEYWORDS

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.timeout_seconds = settings.pipeline.request_timeout_seconds
        self.cache_dir = settings.pipeline.cache_dir
        self.session = requests.Session()

    def get_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        feed_urls = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US",
            f"https://finance.yahoo.com/rss/headline?s={quote_plus(ticker)}",
        ]

        payload = None
        for feed_url in feed_urls:
            try:
                payload = self._get_feed_xml(feed_url, ticker)
            except requests.RequestException as exc:
                LOGGER.warning("Yahoo Finance RSS failed for %s via %s: %s", ticker, feed_url, exc)
                continue
            if payload:
                break

        if not payload:
            return []

        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            LOGGER.warning("Yahoo Finance RSS parse failed for %s: %s", ticker, exc)
            return []

        channel = root.find("channel")
        if channel is None:
            channel = root.find("./rss/channel")
        if channel is None:
            return []

        events: list[CatalystEvent] = []
        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            pub_date = self._parse_rss_timestamp(item.findtext("pubDate"))
            if not title or pub_date is None:
                continue
            if not (start_date <= pub_date.date() <= end_date):
                continue

            combined_text = f"{title} {description}".lower()
            is_regulatory = any(keyword in combined_text for keyword in self.REGULATORY_KEYWORDS)
            source_name = self._extract_source(item)
            events.append(
                CatalystEvent(
                    trade_date=pub_date.date(),
                    ticker=ticker,
                    event_type="fda_decision" if is_regulatory else "company_news",
                    detail=title,
                    source=source_name,
                    source_timestamp=pub_date,
                    as_of_timestamp=pub_date,
                    direction_hint=self._infer_news_direction(combined_text),
                )
            )
        return events

    def _get_feed_xml(self, url: str, ticker: str) -> str | None:
        cache_path = self.cache_dir / "yahoo_finance" / "news" / ticker / f"{date.today().isoformat()}.xml"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.text
        if "<rss" not in payload.lower():
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(payload, encoding="utf-8")
        return payload

    def _extract_source(self, item: ET.Element) -> str:
        source_text = (item.findtext("source") or "").strip()
        if source_text:
            return source_text
        creator_text = item.findtext("{http://purl.org/dc/elements/1.1/}creator")
        if creator_text:
            return creator_text.strip()
        return "Yahoo"

    def _infer_news_direction(self, text: str) -> str:
        positive_hits = sum(1 for keyword in self.POSITIVE_NEWS_KEYWORDS if keyword in text)
        negative_hits = sum(1 for keyword in self.NEGATIVE_NEWS_KEYWORDS if keyword in text)
        if positive_hits > negative_hits:
            return "positive"
        if negative_hits > positive_hits:
            return "negative"
        return "ambiguous"

    @staticmethod
    def _parse_rss_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
        ]
        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None


class GoogleNewsRssCatalystClient:
    """Google News RSS-backed catalyst client with broad source coverage."""

    POSITIVE_NEWS_KEYWORDS = FinnhubCatalystClient.POSITIVE_NEWS_KEYWORDS
    NEGATIVE_NEWS_KEYWORDS = FinnhubCatalystClient.NEGATIVE_NEWS_KEYWORDS
    REGULATORY_KEYWORDS = FinnhubCatalystClient.REGULATORY_KEYWORDS

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.timeout_seconds = settings.pipeline.request_timeout_seconds
        self.cache_dir = settings.pipeline.cache_dir
        self.session = requests.Session()

    def get_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        query = quote_plus(f'"{ticker}" stock')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        try:
            payload = self._get_feed_xml(url, ticker)
        except requests.RequestException as exc:
            LOGGER.warning("Google News RSS failed for %s: %s", ticker, exc)
            return []
        if not payload:
            return []

        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            LOGGER.warning("Google News RSS parse failed for %s: %s", ticker, exc)
            return []

        channel = root.find("channel")
        if channel is None:
            channel = root.find("./rss/channel")
        if channel is None:
            return []

        events: list[CatalystEvent] = []
        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            pub_date = YahooFinanceNewsCatalystClient._parse_rss_timestamp(item.findtext("pubDate"))
            if not title or pub_date is None:
                continue
            if not (start_date <= pub_date.date() <= end_date):
                continue
            combined_text = f"{title} {description}".lower()
            upper_title = title.upper()
            if ticker.upper() not in upper_title and f" {ticker.upper()} " not in combined_text.upper():
                continue

            source_name = self._extract_source(item)
            is_regulatory = any(keyword in combined_text for keyword in self.REGULATORY_KEYWORDS)
            events.append(
                CatalystEvent(
                    trade_date=pub_date.date(),
                    ticker=ticker,
                    event_type="fda_decision" if is_regulatory else "company_news",
                    detail=title,
                    source=source_name,
                    source_timestamp=pub_date,
                    as_of_timestamp=pub_date,
                    direction_hint=self._infer_news_direction(combined_text),
                )
            )
        return events

    def _get_feed_xml(self, url: str, ticker: str) -> str | None:
        cache_path = self.cache_dir / "google_news" / ticker / f"{date.today().isoformat()}.xml"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.text
        if "<rss" not in payload.lower():
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(payload, encoding="utf-8")
        return payload

    @staticmethod
    def _extract_source(item: ET.Element) -> str:
        source_element = item.find("source")
        if source_element is not None and source_element.text:
            return source_element.text.strip()
        return "Google News"

    def _infer_news_direction(self, text: str) -> str:
        positive_hits = sum(1 for keyword in self.POSITIVE_NEWS_KEYWORDS if keyword in text)
        negative_hits = sum(1 for keyword in self.NEGATIVE_NEWS_KEYWORDS if keyword in text)
        if positive_hits > negative_hits:
            return "positive"
        if negative_hits > positive_hits:
            return "negative"
        return "ambiguous"


class CombinedNewsCatalystClient:
    """Combine low-friction news feeds without requiring Finnhub."""

    def __init__(self, clients: list[CatalystClient]) -> None:
        self.clients = clients

    def get_events(self, ticker: str, start_date: date, end_date: date) -> Iterable[CatalystEvent]:
        seen: set[tuple[str, str, str, str]] = set()
        for client in self.clients:
            for event in client.get_events(ticker, start_date, end_date):
                key = (
                    event.trade_date.isoformat(),
                    event.event_type,
                    event.detail,
                    event.source_timestamp.isoformat(),
                )
                if key in seen:
                    continue
                seen.add(key)
                yield event


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
    selected_sources = {source.strip().lower() for source in settings.catalysts.sources if source.strip()}
    news_clients: list[CatalystClient] = []

    if "google_news" in selected_sources and settings.providers.google_news.enabled:
        news_clients.append(GoogleNewsRssCatalystClient(settings))
    if "yahoo_finance" in selected_sources and settings.providers.yahoo_finance.enabled:
        news_clients.append(YahooFinanceNewsCatalystClient(settings))

    client: CatalystClient
    if news_clients:
        client = CombinedNewsCatalystClient(news_clients)
    elif settings.pipeline.use_stub_fallback:
        LOGGER.warning("No live catalyst news sources are enabled; falling back to stub catalyst client")
        client = StubCatalystClient()
    else:
        raise ValueError("At least one live catalyst source must be enabled when stub fallback is disabled")
    loader = CatalystLoader(settings=settings, db=db, client=client)
    return loader.load_range(start_date=start_date, end_date=end_date, tickers=tickers)
