"""Point-in-time timestamp helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Iterator


def ensure_utc_timestamp(value: datetime, label: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be in UTC")


def trading_days(start_date: date, end_date: date) -> Iterator[date]:
    cursor = start_date
    while cursor <= end_date:
        if cursor.weekday() < 5:
            yield cursor
        cursor += timedelta(days=1)


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)
