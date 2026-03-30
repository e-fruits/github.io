"""End-of-day options flow aggregation."""

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
class OptionContractSnapshot:
    trade_date: date
    ticker: str
    option_type: str
    volume: float
    open_interest: float
    small_lot_volume: float | None
    as_of_timestamp: datetime


class OptionsFlowClient(Protocol):
    def get_chain_snapshot(self, ticker: str, trade_date: date) -> Iterable[OptionContractSnapshot]:
        """Return end-of-day option chain snapshot for a ticker/day."""


class StubOptionsFlowClient:
    def get_chain_snapshot(self, ticker: str, trade_date: date) -> Iterable[OptionContractSnapshot]:
        base_time = datetime.combine(trade_date, time(hour=21, minute=0), tzinfo=timezone.utc)
        if ticker == "NOOPT":
            return []
        factor = (trade_date.day % 5) + 1
        return [
            OptionContractSnapshot(
                trade_date=trade_date,
                ticker=ticker,
                option_type="call",
                volume=float(100 * factor),
                open_interest=float(500 + factor * 30),
                small_lot_volume=float(40 * factor),
                as_of_timestamp=base_time,
            ),
            OptionContractSnapshot(
                trade_date=trade_date,
                ticker=ticker,
                option_type="put",
                volume=float(55 * factor),
                open_interest=float(320 + factor * 15),
                small_lot_volume=float(18 * factor),
                as_of_timestamp=base_time,
            ),
        ]


class OptionsFlowLoader:
    def __init__(self, settings: AppSettings, db: DatabaseManager, client: OptionsFlowClient) -> None:
        self.settings = settings
        self.db = db
        self.client = client

    def load_range(self, start_date: date, end_date: date, tickers: Iterable[str] | None = None) -> int:
        if tickers is None:
            tickers = self.db.fetch_universe_tickers(start_date.isoformat(), end_date.isoformat())

        rows: list[dict[str, object]] = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                for ticker in tickers:
                    rows.append(self._aggregate_day(ticker, current))
            current += timedelta(days=1)

        enriched = self._compute_rollups(pd.DataFrame(rows))
        payload = [self._row_to_record(row) for row in enriched.to_dict(orient="records")]
        self.db.upsert_options_flow_daily(payload)
        LOGGER.info("Stored options flow rows=%s", len(payload))
        return len(payload)

    def _aggregate_day(self, ticker: str, trade_date: date) -> dict[str, object]:
        snapshot = list(self.client.get_chain_snapshot(ticker, trade_date))
        if not snapshot:
            as_of_timestamp = datetime.combine(trade_date, time(hour=21, minute=0), tzinfo=timezone.utc)
            return {
                "date": pd.Timestamp(trade_date),
                "ticker": ticker,
                "total_call_volume": 0.0,
                "total_put_volume": 0.0,
                "total_call_oi": 0.0,
                "total_put_oi": 0.0,
                "small_lot_call_volume": None,
                "small_lot_put_volume": None,
                "as_of_timestamp": as_of_timestamp,
            }

        for contract in snapshot:
            ensure_utc_timestamp(contract.as_of_timestamp, label=f"{ticker}.{trade_date}.options_as_of")

        calls = [contract for contract in snapshot if contract.option_type == "call"]
        puts = [contract for contract in snapshot if contract.option_type == "put"]
        as_of_timestamp = max(contract.as_of_timestamp for contract in snapshot)
        return {
            "date": pd.Timestamp(trade_date),
            "ticker": ticker,
            "total_call_volume": float(sum(contract.volume for contract in calls)),
            "total_put_volume": float(sum(contract.volume for contract in puts)),
            "total_call_oi": float(sum(contract.open_interest for contract in calls)),
            "total_put_oi": float(sum(contract.open_interest for contract in puts)),
            "small_lot_call_volume": self._sum_optional(contract.small_lot_volume for contract in calls),
            "small_lot_put_volume": self._sum_optional(contract.small_lot_volume for contract in puts),
            "as_of_timestamp": as_of_timestamp,
        }

    def _compute_rollups(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        frame["call_put_ratio"] = frame["total_call_volume"] / frame["total_put_volume"].replace(0.0, pd.NA)
        frame["oi_change_calls"] = frame.groupby("ticker")["total_call_oi"].diff()
        frame["oi_change_puts"] = frame.groupby("ticker")["total_put_oi"].diff()
        frame["total_options_volume"] = frame["total_call_volume"] + frame["total_put_volume"]
        baseline = frame.groupby("ticker")["total_options_volume"].transform(
            lambda series: series.shift(1).rolling(
                window=self.settings.options_flow.rolling_window_days,
                min_periods=self.settings.options_flow.rolling_window_days,
            ).mean()
        )
        frame["unusual_volume_ratio"] = frame["total_options_volume"] / baseline
        total_small_lot = frame["small_lot_call_volume"].fillna(0.0) + frame["small_lot_put_volume"].fillna(0.0)
        frame["small_lot_call_pct"] = frame["small_lot_call_volume"] / frame["total_call_volume"].replace(0.0, pd.NA)
        frame["total_small_lot_pct"] = total_small_lot / frame["total_options_volume"].replace(0.0, pd.NA)
        return frame

    @staticmethod
    def _row_to_record(row: dict[str, object]) -> dict[str, object]:
        return {
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "ticker": row["ticker"],
            "total_call_volume": OptionsFlowLoader._to_float_or_none(row.get("total_call_volume")),
            "total_put_volume": OptionsFlowLoader._to_float_or_none(row.get("total_put_volume")),
            "call_put_ratio": OptionsFlowLoader._to_float_or_none(row.get("call_put_ratio")),
            "total_call_oi": OptionsFlowLoader._to_float_or_none(row.get("total_call_oi")),
            "total_put_oi": OptionsFlowLoader._to_float_or_none(row.get("total_put_oi")),
            "oi_change_calls": OptionsFlowLoader._to_float_or_none(row.get("oi_change_calls")),
            "oi_change_puts": OptionsFlowLoader._to_float_or_none(row.get("oi_change_puts")),
            "unusual_volume_ratio": OptionsFlowLoader._to_float_or_none(row.get("unusual_volume_ratio")),
            "small_lot_call_volume": OptionsFlowLoader._to_float_or_none(row.get("small_lot_call_volume")),
            "small_lot_put_volume": OptionsFlowLoader._to_float_or_none(row.get("small_lot_put_volume")),
            "small_lot_call_pct": OptionsFlowLoader._to_float_or_none(row.get("small_lot_call_pct")),
            "as_of_timestamp": row["as_of_timestamp"].isoformat(),
        }

    @staticmethod
    def _sum_optional(values: Iterable[float | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return float(sum(filtered))

    @staticmethod
    def _to_float_or_none(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)


def load_options_flow_from_config(
    start_date: date,
    end_date: date,
    settings_path: Path = Path("config/settings.yaml"),
    tickers: Iterable[str] | None = None,
) -> int:
    settings = load_settings(settings_path)
    db = DatabaseManager(settings.database.path)
    db.initialize()
    loader = OptionsFlowLoader(settings=settings, db=db, client=StubOptionsFlowClient())
    return loader.load_range(start_date=start_date, end_date=end_date, tickers=tickers)
