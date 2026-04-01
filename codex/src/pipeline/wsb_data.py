"""WSB / Reddit mention ingestion with ApeWisdom, PRAW, and Kaggle paths."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd
import requests

from src.utils.config import AppSettings, load_settings
from src.utils.db import DatabaseManager
from src.utils.http import HttpRequestError, JsonHttpClient
from src.utils.timestamps import ensure_utc_timestamp

LOGGER = logging.getLogger(__name__)
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
STOPWORDS = {"A", "I", "AI", "IT", "ALL", "FOR", "YOLO", "WSB", "USA"}


@dataclass
class ApeWisdomMention:
    trade_date: date
    ticker: str
    mention_count: float
    upvotes: float
    rank: float
    rank_change_24h: float | None
    as_of_timestamp: datetime


@dataclass
class RedditMessage:
    created_at: datetime
    ticker: str
    text: str
    score: float
    as_of_timestamp: datetime


class ApeWisdomClient(Protocol):
    def get_daily_top_mentions(self, trade_date: date, top_n: int) -> Iterable[ApeWisdomMention]:
        """Return daily top-mentioned tickers from ApeWisdom."""


class RedditMessageClient(Protocol):
    def get_messages(self, start_date: date, end_date: date, subreddit: str) -> Iterable[RedditMessage]:
        """Return WSB posts/comments with timestamps and message text."""


class StubApeWisdomClient:
    def get_daily_top_mentions(self, trade_date: date, top_n: int) -> Iterable[ApeWisdomMention]:
        del top_n
        base_timestamp = datetime.combine(trade_date, time(hour=13, minute=0), tzinfo=timezone.utc)
        day_factor = trade_date.day % 7
        return [
            ApeWisdomMention(
                trade_date=trade_date,
                ticker="ABCD",
                mention_count=float(35 + day_factor * 4),
                upvotes=float(1200 + day_factor * 100),
                rank=1.0,
                rank_change_24h=float(-1 + (day_factor % 3)),
                as_of_timestamp=base_timestamp,
            ),
            ApeWisdomMention(
                trade_date=trade_date,
                ticker="EFGH",
                mention_count=float(20 + day_factor * 2),
                upvotes=float(800 + day_factor * 80),
                rank=2.0,
                rank_change_24h=float(day_factor % 2),
                as_of_timestamp=base_timestamp,
            ),
        ]


class StubRedditMessageClient:
    def get_messages(self, start_date: date, end_date: date, subreddit: str) -> Iterable[RedditMessage]:
        del subreddit
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                ts = datetime.combine(current, time(hour=22, minute=0), tzinfo=timezone.utc)
                yield RedditMessage(
                    created_at=ts,
                    ticker="ABCD",
                    text="ABCD looks bullish, huge beat, strong breakout and squeeze setup",
                    score=float(250 + current.day * 3),
                    as_of_timestamp=ts + timedelta(minutes=5),
                )
                yield RedditMessage(
                    created_at=ts + timedelta(minutes=10),
                    ticker="ABCD",
                    text="ABCD still bullish and ripping",
                    score=float(180 + current.day * 2),
                    as_of_timestamp=ts + timedelta(minutes=15),
                )
                yield RedditMessage(
                    created_at=ts + timedelta(minutes=20),
                    ticker="EFGH",
                    text="EFGH looks weak and bearish after dilution",
                    score=float(90 + current.day),
                    as_of_timestamp=ts + timedelta(minutes=25),
                )
            current += timedelta(days=1)


class RealApeWisdomClient:
    def __init__(self, settings: AppSettings) -> None:
        self.http = JsonHttpClient(
            base_url=settings.providers.apewisdom.base_url,
            cache_dir=settings.pipeline.cache_dir,
            timeout_seconds=settings.pipeline.request_timeout_seconds,
        )

    def get_daily_top_mentions(self, trade_date: date, top_n: int) -> Iterable[ApeWisdomMention]:
        payload = self.http.get_json(
            f"/filter/all-stocks/page/1",
            cache_namespace=f"apewisdom/daily/{trade_date.isoformat()}",
        )
        rows = payload.get("results", payload.get("data", []))
        if not isinstance(rows, list):
            return []

        as_of_timestamp = datetime.combine(trade_date, time(hour=13, minute=0), tzinfo=timezone.utc)
        mentions: list[ApeWisdomMention] = []
        for rank_index, row in enumerate(rows[:top_n], start=1):
            if not isinstance(row, dict) or not row.get("ticker"):
                continue
            mentions.append(
                ApeWisdomMention(
                    trade_date=trade_date,
                    ticker=str(row["ticker"]).upper(),
                    mention_count=float(row.get("mentions") or row.get("mention_count") or 0.0),
                    upvotes=float(row.get("upvotes") or row.get("total_upvotes") or 0.0),
                    rank=float(row.get("rank") or rank_index),
                    rank_change_24h=self._to_float(row.get("rank_24h_ago") or row.get("rank_change_24h")),
                    as_of_timestamp=as_of_timestamp,
                )
            )
        return mentions

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class RedditOauthClient:
    SEARCH_LIMIT = 100

    def __init__(self, settings: AppSettings) -> None:
        credentials = settings.providers.reddit
        if not credentials.client_id or not credentials.client_secret:
            raise ValueError("Reddit client_id/client_secret are required for the live Reddit client")
        self.settings = settings
        self.credentials = credentials
        self.cache_dir = settings.pipeline.cache_dir
        self.timeout_seconds = settings.pipeline.request_timeout_seconds
        self._token: str | None = None
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": credentials.user_agent})

    def get_messages(self, start_date: date, end_date: date, subreddit: str) -> Iterable[RedditMessage]:
        trade_dates = self._trading_dates(start_date, end_date)
        if not trade_dates:
            return []

        token = self._get_token()
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": self.credentials.user_agent,
            }
        )

        messages: list[RedditMessage] = []
        for trade_date in trade_dates:
            query = f"timestamp:{int(datetime.combine(trade_date, time.min, tzinfo=timezone.utc).timestamp())}..{int(datetime.combine(trade_date, time.max, tzinfo=timezone.utc).timestamp())}"
            try:
                response = session.get(
                    f"https://oauth.reddit.com/r/{subreddit}/search.json",
                    params={
                        "q": query,
                        "restrict_sr": "on",
                        "sort": "new",
                        "syntax": "cloudsearch",
                        "limit": self.SEARCH_LIMIT,
                        "t": "all",
                    },
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                LOGGER.warning("Reddit search failed for %s on %s: %s", subreddit, trade_date, exc)
                continue

            payload = response.json()
            children = payload.get("data", {}).get("children", [])
            for child in children:
                post_data = child.get("data", {})
                created_ts = post_data.get("created_utc")
                body = post_data.get("selftext") or post_data.get("title") or ""
                created_at = None if created_ts is None else datetime.fromtimestamp(float(created_ts), tz=timezone.utc)
                if created_at is None or created_at.date() != trade_date:
                    continue
                ticker = WsbDataLoader._extract_primary_ticker(str(body))
                if ticker is None:
                    continue
                messages.append(
                    RedditMessage(
                        created_at=created_at,
                        ticker=ticker,
                        text=str(body),
                        score=float(post_data.get("score") or 0.0),
                        as_of_timestamp=created_at + timedelta(minutes=1),
                    )
                )
        return messages

    def _get_token(self) -> str:
        if self._token is not None:
            return self._token
        try:
            response = self._session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(self.credentials.client_id, self.credentials.client_secret),
                data={"grant_type": "client_credentials"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Unable to authenticate with Reddit: {exc}") from exc
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise RuntimeError("Reddit authentication returned no access token")
        self._token = str(token)
        return self._token

    @staticmethod
    def _trading_dates(start_date: date, end_date: date) -> list[date]:
        dates: list[date] = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)
        return dates


class SimpleSentimentAnalyzer:
    """Use optional FinBERT/VADER when installed, otherwise fall back to keyword scoring."""

    POSITIVE_WORDS = {"bullish", "beat", "strong", "upgrade", "buy", "squeeze", "rip", "ripping", "breakout"}
    NEGATIVE_WORDS = {"bearish", "miss", "weak", "downgrade", "sell", "dilution", "dump", "rug"}

    def __init__(self, preferred_model: str = "finbert", fallback_model: str = "vader") -> None:
        self.preferred_model = preferred_model
        self.fallback_model = fallback_model
        self._transformer = self._try_load_transformer() if preferred_model == "finbert" else None
        self._vader = self._try_load_vader() if self._transformer is None and fallback_model == "vader" else None

    def score(self, text: str) -> float:
        if self._transformer is not None:
            result = self._transformer(text[:512])[0]
            label = str(result["label"]).lower()
            score = float(result["score"])
            if "positive" in label:
                return score
            if "negative" in label:
                return -score
            return 0.0

        if self._vader is not None:
            return float(self._vader.polarity_scores(text)["compound"])

        tokens = {token.lower() for token in re.findall(r"[A-Za-z']+", text)}
        positive_hits = len(tokens & self.POSITIVE_WORDS)
        negative_hits = len(tokens & self.NEGATIVE_WORDS)
        total = positive_hits + negative_hits
        if total == 0:
            return 0.0
        return (positive_hits - negative_hits) / total

    @staticmethod
    def _try_load_transformer():
        try:
            from transformers import pipeline
        except Exception:
            return None
        try:
            return pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception:
            return None

    @staticmethod
    def _try_load_vader():
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
        except Exception:
            return None
        try:
            return SentimentIntensityAnalyzer()
        except Exception:
            return None


class WsbDataLoader:
    def __init__(
        self,
        settings: AppSettings,
        db: DatabaseManager,
        apewisdom_client: ApeWisdomClient | None = None,
        reddit_client: RedditMessageClient | None = None,
        sentiment_analyzer: SimpleSentimentAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.apewisdom_client = apewisdom_client or self._default_apewisdom_client(settings)
        self.reddit_client = reddit_client or self._default_reddit_client(settings)
        self.sentiment_analyzer = sentiment_analyzer or SimpleSentimentAnalyzer(
            preferred_model=settings.wsb.sentiment_model,
            fallback_model=settings.wsb.fallback_sentiment_model,
        )

    def load_apewisdom_range(self, start_date: date, end_date: date) -> int:
        rows: list[dict[str, object]] = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                for mention in self.apewisdom_client.get_daily_top_mentions(current, self.settings.wsb.apewisdom_top_n):
                    ensure_utc_timestamp(mention.as_of_timestamp, label=f"{mention.ticker}.{current}.apewisdom_as_of")
                    rows.append(
                        {
                            "date": mention.trade_date.isoformat(),
                            "ticker": mention.ticker,
                            "mention_count": mention.mention_count,
                            "mention_count_prior_day": None,
                            "mention_velocity_pct": None,
                            "mention_vs_baseline": None,
                            "sentiment_score": None,
                            "sentiment_unanimity": None,
                            "upvotes": mention.upvotes,
                            "rank": mention.rank,
                            "rank_change_24h": mention.rank_change_24h,
                            "as_of_timestamp": mention.as_of_timestamp.isoformat(),
                            "source": "apewisdom",
                        }
                    )
            current += timedelta(days=1)

        normalized = self._compute_rollups(pd.DataFrame(rows), source="apewisdom")
        self.db.upsert_wsb_mentions(normalized)
        LOGGER.info("Stored ApeWisdom rows=%s", len(normalized))
        return len(normalized)

    def load_praw_range(self, start_date: date, end_date: date) -> int:
        messages = list(self.reddit_client.get_messages(start_date, end_date, self.settings.wsb.subreddit))
        if not messages:
            return 0

        rows: list[dict[str, object]] = []
        grouped: dict[tuple[date, str], list[RedditMessage]] = {}
        for message in messages:
            ensure_utc_timestamp(message.created_at, label=f"{message.ticker}.created_at")
            ensure_utc_timestamp(message.as_of_timestamp, label=f"{message.ticker}.as_of_timestamp")
            grouped.setdefault((message.created_at.date(), message.ticker), []).append(message)

        for (trade_date, ticker), bucket in grouped.items():
            scores = [self.sentiment_analyzer.score(message.text) for message in bucket]
            dominant_count = sum(1 for score in scores if self._label(score) == self._dominant_label(scores))
            total_labeled = sum(1 for score in scores if self._label(score) != "neutral")
            sentiment_unanimity = None if total_labeled == 0 else dominant_count / total_labeled
            average_score = None if not scores else float(sum(scores) / len(scores))
            upvotes = float(sum(message.score for message in bucket))
            as_of_timestamp = max(message.as_of_timestamp for message in bucket)
            rows.append(
                {
                    "date": trade_date.isoformat(),
                    "ticker": ticker,
                    "mention_count": float(len(bucket)),
                    "mention_count_prior_day": None,
                    "mention_velocity_pct": None,
                    "mention_vs_baseline": None,
                    "sentiment_score": average_score,
                    "sentiment_unanimity": sentiment_unanimity,
                    "upvotes": upvotes,
                    "rank": None,
                    "rank_change_24h": None,
                    "as_of_timestamp": as_of_timestamp.isoformat(),
                    "source": "praw",
                }
            )

        normalized = self._compute_rollups(pd.DataFrame(rows), source="praw")
        self.db.upsert_wsb_mentions(normalized)
        LOGGER.info("Stored PRAW rows=%s", len(normalized))
        return len(normalized)

    def load_kaggle_history(self, dataset_path: Path | None = None) -> int:
        source_path = dataset_path or self.settings.wsb.kaggle_dataset_path
        if not source_path.exists():
            LOGGER.warning("Kaggle WSB dataset not found at %s", source_path)
            return 0

        frame = pd.read_csv(source_path)
        if "date" not in frame.columns:
            raise ValueError("Kaggle WSB dataset must contain a date column")

        normalized = self._normalize_kaggle_frame(frame)
        rolled = self._compute_rollups(normalized, source="kaggle")
        self.db.upsert_wsb_mentions(rolled)
        LOGGER.info("Stored Kaggle WSB rows=%s from %s", len(rolled), source_path)
        return len(rolled)

    def _normalize_kaggle_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.date.astype(str)

        if "ticker" not in working.columns:
            text_column = "body" if "body" in working.columns else "text"
            if text_column not in working.columns:
                raise ValueError("Kaggle WSB dataset must contain ticker or body/text")
            working["ticker"] = working[text_column].astype(str).apply(self._extract_primary_ticker)

        if "mention_count" not in working.columns:
            working["mention_count"] = 1.0
        if "upvotes" not in working.columns:
            score_column = "score" if "score" in working.columns else None
            working["upvotes"] = 0.0 if score_column is None else working[score_column].fillna(0.0)
        if "sentiment_score" not in working.columns:
            text_column = "body" if "body" in working.columns else "text"
            if text_column in working.columns:
                working["sentiment_score"] = working[text_column].astype(str).apply(self.sentiment_analyzer.score)
            else:
                working["sentiment_score"] = None

        grouped = (
            working.dropna(subset=["ticker"])
            .groupby(["date", "ticker"], as_index=False)
            .agg(
                mention_count=("mention_count", "sum"),
                upvotes=("upvotes", "sum"),
                sentiment_score=("sentiment_score", "mean"),
            )
        )
        grouped["sentiment_unanimity"] = None
        grouped["rank"] = None
        grouped["rank_change_24h"] = None
        grouped["as_of_timestamp"] = grouped["date"].apply(lambda value: f"{value}T23:59:00+00:00")
        grouped["source"] = "kaggle"
        grouped["mention_count_prior_day"] = None
        grouped["mention_velocity_pct"] = None
        grouped["mention_vs_baseline"] = None
        return grouped

    def _compute_rollups(self, frame: pd.DataFrame, source: str) -> list[dict[str, object]]:
        if frame.empty:
            return []

        working = frame.copy()
        working["date"] = pd.to_datetime(working["date"])
        working = working.sort_values(["ticker", "date"]).reset_index(drop=True)
        rolling_window = self.settings.wsb.rolling_window_days
        working["mention_count_prior_day"] = working.groupby("ticker")["mention_count"].shift(1)
        working["mention_velocity_pct"] = (
            (working["mention_count"] - working["mention_count_prior_day"]) / working["mention_count_prior_day"]
        )
        baseline = working.groupby("ticker")["mention_count"].transform(
            lambda series: series.shift(1).rolling(window=rolling_window, min_periods=rolling_window).mean()
        )
        working["mention_vs_baseline"] = working["mention_count"] / baseline
        working["source"] = source

        result: list[dict[str, object]] = []
        for row in working.to_dict(orient="records"):
            result.append(
                {
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "ticker": row["ticker"],
                    "mention_count": self._to_float_or_none(row["mention_count"]),
                    "mention_count_prior_day": self._to_float_or_none(row.get("mention_count_prior_day")),
                    "mention_velocity_pct": self._to_float_or_none(row.get("mention_velocity_pct")),
                    "mention_vs_baseline": self._to_float_or_none(row.get("mention_vs_baseline")),
                    "sentiment_score": self._to_float_or_none(row.get("sentiment_score")),
                    "sentiment_unanimity": self._to_float_or_none(row.get("sentiment_unanimity")),
                    "upvotes": self._to_float_or_none(row.get("upvotes")),
                    "rank": self._to_float_or_none(row.get("rank")),
                    "rank_change_24h": self._to_float_or_none(row.get("rank_change_24h")),
                    "as_of_timestamp": str(row["as_of_timestamp"]),
                    "source": source,
                }
            )
        return result

    @staticmethod
    def _default_apewisdom_client(settings: AppSettings) -> ApeWisdomClient:
        try:
            return RealApeWisdomClient(settings)
        except Exception as exc:
            if not settings.pipeline.use_stub_fallback:
                raise
            LOGGER.warning("Falling back to stub ApeWisdom client: %s", exc)
            return StubApeWisdomClient()

    @staticmethod
    def _default_reddit_client(settings: AppSettings) -> RedditMessageClient:
        credentials = settings.providers.reddit
        if credentials.client_id and credentials.client_secret:
            try:
                return RedditOauthClient(settings)
            except Exception as exc:
                if not settings.pipeline.use_stub_fallback:
                    raise
                LOGGER.warning("Falling back to stub Reddit client: %s", exc)
        elif not settings.pipeline.use_stub_fallback:
            raise ValueError("Reddit credentials are required when stub fallback is disabled")
        else:
            LOGGER.warning("Reddit credentials not configured; falling back to stub Reddit client")
        return StubRedditMessageClient()

    @staticmethod
    def _dominant_label(scores: list[float]) -> str:
        labels = [WsbDataLoader._label(score) for score in scores if WsbDataLoader._label(score) != "neutral"]
        if not labels:
            return "neutral"
        positive_count = labels.count("positive")
        negative_count = labels.count("negative")
        return "positive" if positive_count >= negative_count else "negative"

    @staticmethod
    def _label(score: float) -> str:
        if score > 0.05:
            return "positive"
        if score < -0.05:
            return "negative"
        return "neutral"

    @staticmethod
    def _extract_primary_ticker(text: str) -> str | None:
        matches = [match for match in TICKER_PATTERN.findall(text.upper()) if match not in STOPWORDS]
        return matches[0] if matches else None

    @staticmethod
    def _to_float_or_none(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)


def load_wsb_from_config(
    start_date: date,
    end_date: date,
    settings_path: Path = Path("config/settings.yaml"),
) -> int:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    loader = WsbDataLoader(settings=settings, db=db)
    total = 0
    if "apewisdom" in settings.wsb.sources:
        total += loader.load_apewisdom_range(start_date, end_date)
    if "praw" in settings.wsb.sources:
        total += loader.load_praw_range(start_date, end_date)
    if "kaggle" in settings.wsb.sources:
        total += loader.load_kaggle_history()
    return total
