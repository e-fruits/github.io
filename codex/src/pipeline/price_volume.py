"""Historical daily OHLCV ingestion and derived metric computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from src.utils.config import AppSettings, load_settings
from src.utils.db import DatabaseManager
from src.utils.timestamps import ensure_utc_timestamp

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DailyBar:
    trade_date: date
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None
    as_of_timestamp: datetime


@dataclass(slots=True)
class PremarketSnapshot:
    trade_date: date
    ticker: str
    high: float | None
    low: float | None
    volume: float | None
    dollar_volume: float | None
    as_of_timestamp: datetime


class HistoricalPriceClient(Protocol):
    def get_daily_bars(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        adjusted: bool = True,
    ) -> Iterable[DailyBar]:
        """Return historical daily bars for one ticker."""

    def get_premarket_snapshot(
        self,
        ticker: str,
        trade_date: date,
    ) -> PremarketSnapshot | None:
        """Return premarket summary known before the open for a given ticker/day."""


class StubHistoricalPriceClient:
    """Synthetic client to keep the loader runnable before real Polygon integration."""

    def get_daily_bars(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        adjusted: bool = True,
    ) -> Iterable[DailyBar]:
        del adjusted
        cursor = start_date
        offset = sum(ord(char) for char in ticker) % 7
        seed_price = 10.0 + offset

        while cursor <= end_date:
            if cursor.weekday() < 5:
                day_index = (cursor - start_date).days + 1
                base = seed_price + day_index * 0.15
                open_price = round(base, 2)
                close_price = round(base * (1.0 + ((day_index % 5) - 2) * 0.004), 2)
                high_price = round(max(open_price, close_price) * 1.02, 2)
                low_price = round(min(open_price, close_price) * 0.98, 2)
                volume = float(900_000 + (day_index % 10) * 80_000 + offset * 25_000)
                as_of_timestamp = datetime.combine(cursor, time(hour=20, minute=0), tzinfo=timezone.utc)
                yield DailyBar(
                    trade_date=cursor,
                    ticker=ticker,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    vwap=round((open_price + high_price + low_price + close_price) / 4.0, 2),
                    as_of_timestamp=as_of_timestamp,
                )
            cursor += timedelta(days=1)

    def get_premarket_snapshot(self, ticker: str, trade_date: date) -> PremarketSnapshot | None:
        if trade_date.weekday() >= 5:
            return None
        base = 11.0 + (sum(ord(char) for char in ticker) % 5)
        volume = float(60_000 + trade_date.day * 2_000)
        return PremarketSnapshot(
            trade_date=trade_date,
            ticker=ticker,
            high=round(base * 1.03, 2),
            low=round(base * 0.99, 2),
            volume=volume,
            dollar_volume=round(volume * base, 2),
            as_of_timestamp=datetime.combine(trade_date, time(hour=13, minute=25), tzinfo=timezone.utc),
        )


class PriceVolumeLoader:
    def __init__(self, settings: AppSettings, db: DatabaseManager, client: HistoricalPriceClient) -> None:
        self.settings = settings
        self.db = db
        self.client = client

    def load_range(self, start_date: date, end_date: date, tickers: Iterable[str] | None = None) -> int:
        if tickers is None:
            tickers = self.db.fetch_universe_tickers(start_date.isoformat(), end_date.isoformat())

        total_rows = 0
        for ticker in tickers:
            total_rows += self.load_ticker(ticker, start_date, end_date)

        LOGGER.info("Stored daily price rows=%s from %s to %s", total_rows, start_date, end_date)
        return total_rows

    def load_ticker(self, ticker: str, start_date: date, end_date: date) -> int:
        lookback_days = max(self.settings.price_volume.rolling_window_days * 3, 40)
        history_start = start_date - timedelta(days=lookback_days)
        bars = list(
            self.client.get_daily_bars(
                ticker=ticker,
                start_date=history_start,
                end_date=end_date,
                adjusted=self.settings.price_volume.adjusted,
            )
        )
        if not bars:
            LOGGER.warning("No daily bars returned for %s", ticker)
            return 0

        frame = self._compute_metrics(bars)
        frame = frame[(frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))]

        rows: list[dict[str, object]] = []
        for row in frame.to_dict(orient="records"):
            trade_date = row["date"].date()
            premarket = self.client.get_premarket_snapshot(ticker, trade_date) if self.settings.price_volume.include_premarket else None
            if premarket is not None:
                ensure_utc_timestamp(premarket.as_of_timestamp, label=f"{ticker}.{trade_date}.premarket_as_of")

            avg_volume = row["avg_volume_20d"]
            premarket_rel_volume = None
            if premarket is not None and avg_volume is not None and not pd.isna(avg_volume) and float(avg_volume) != 0.0:
                premarket_rel_volume = float(premarket.volume or 0.0) / float(avg_volume)

            rows.append(
                {
                    "date": trade_date.isoformat(),
                    "ticker": ticker,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "vwap": self._to_float_or_none(row["vwap"]),
                    "dollar_volume": self._to_float_or_none(row["dollar_volume"]),
                    "relative_volume_20d": self._to_float_or_none(row["relative_volume_20d"]),
                    "gap_pct": self._to_float_or_none(row["gap_pct"]),
                    "intraday_range_pct": self._to_float_or_none(row["intraday_range_pct"]),
                    "pre_market_high": None if premarket is None else premarket.high,
                    "pre_market_low": None if premarket is None else premarket.low,
                    "pre_market_volume": None if premarket is None else premarket.volume,
                    "pre_market_dollar_volume": None if premarket is None else premarket.dollar_volume,
                    "pre_market_relative_volume": premarket_rel_volume,
                    "as_of_timestamp": row["as_of_timestamp"].isoformat(),
                }
            )

        self.db.upsert_daily_prices(rows)
        LOGGER.info("Stored %s daily price rows for %s", len(rows), ticker)
        return len(rows)

    def _compute_metrics(self, bars: list[DailyBar]) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for bar in bars:
            ensure_utc_timestamp(bar.as_of_timestamp, label=f"{bar.ticker}.{bar.trade_date}.as_of_timestamp")
            records.append(
                {
                    "date": pd.Timestamp(bar.trade_date),
                    "ticker": bar.ticker,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                    "as_of_timestamp": bar.as_of_timestamp,
                }
            )

        frame = pd.DataFrame(records).sort_values(["ticker", "date"]).reset_index(drop=True)
        frame["dollar_volume"] = frame["close"] * frame["volume"]
        window = self.settings.price_volume.rolling_window_days

        # Shift first so the current session cannot use the current day's completed bar.
        frame["avg_volume_20d"] = frame.groupby("ticker")["volume"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=window).mean()
        )
        frame["avg_dollar_volume_20d"] = frame.groupby("ticker")["dollar_volume"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=window).mean()
        )
        frame["prior_close"] = frame.groupby("ticker")["close"].shift(1)
        frame["relative_volume_20d"] = frame["volume"] / frame["avg_volume_20d"]
        frame["gap_pct"] = (frame["open"] - frame["prior_close"]) / frame["prior_close"]
        frame["intraday_range_pct"] = (frame["high"] - frame["low"]) / frame["open"]
        return frame

    @staticmethod
    def _to_float_or_none(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)


def load_daily_prices_from_config(
    start_date: date,
    end_date: date,
    settings_path: Path = Path("config/settings.yaml"),
    tickers: Iterable[str] | None = None,
) -> int:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    loader = PriceVolumeLoader(settings=settings, db=db, client=StubHistoricalPriceClient())
    return loader.load_range(start_date=start_date, end_date=end_date, tickers=tickers)
