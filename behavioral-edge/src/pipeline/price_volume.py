"""
src/pipeline/price_volume.py

Historical OHLCV ingestion from Polygon.io for every ticker in the
point-in-time universe.  Pre-market data (where available) is also
fetched from Polygon's extended-hours aggregates.

Derived fields computed here
----------------------------
  relative_volume   — day volume / rolling 20-day avg volume
  gap_pct           — (open - prior_close) / prior_close
  intraday_range_pct — (high - low) / open

Output tables
-------------
  daily_prices  (see db.py for schema)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.db import fetch_df, get_connection, get_db, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)

_POLYGON_BASE = settings["api"]["polygon"]["base_url"]
_API_KEY = settings["api"]["polygon"]["api_key"]
_RATE_LIMIT = settings["api"]["polygon"]["rate_limit_per_minute"]
_REQUEST_GAP = 60.0 / max(_RATE_LIMIT, 1)


# ── HTTP ───────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(endpoint: str, params: dict | None = None) -> dict:
    params = params or {}
    params["apiKey"] = _API_KEY
    resp = requests.get(
        f"{_POLYGON_BASE}{endpoint}",
        params=params,
        timeout=settings["pipeline"]["request_timeout"],
    )
    resp.raise_for_status()
    time.sleep(_REQUEST_GAP)
    return resp.json()


# ── Polygon data fetchers ──────────────────────────────────────────────────────

def fetch_daily_bars(
    ticker: str,
    from_date: str,
    to_date: str,
    adjusted: bool = True,
) -> list[dict]:
    """
    Fetch daily OHLCV bars from Polygon /v2/aggs for *ticker*.
    Returns a list of bar dicts sorted ascending by date.
    """
    try:
        data = _get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
            {
                "adjusted": str(adjusted).lower(),
                "sort": "asc",
                "limit": 5000,
            },
        )
        return data.get("results", [])
    except Exception as exc:
        log.warning("daily_bars_failed", ticker=ticker, error=str(exc))
        return []


def fetch_premarket_bars(ticker: str, target_date: str) -> dict | None:
    """
    Fetch pre-market (4:00–9:30 ET) aggregate bar for *target_date*.
    Uses Polygon's minute-level aggregates and rolls them up.
    Returns a dict with keys: pre_market_high, pre_market_low, pre_market_volume.
    """
    # Pre-market is 04:00–09:30 ET.  We pull minute bars and aggregate.
    try:
        data = _get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{target_date}/{target_date}",
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": 1000,
                "extended_hours": "true",
            },
        )
        bars = data.get("results", [])
        if not bars:
            return None

        # Polygon timestamps are milliseconds UTC.
        # Pre-market = 08:00–13:30 UTC (= 04:00–09:30 ET during EST, shift 1h for EDT)
        # Use 08:00–13:30 UTC as a rough cut; exact timezone handling can be refined.
        pm_bars = [
            b for b in bars
            if b.get("t") is not None
            and (b["t"] % 86_400_000) < (9 * 3600 + 30 * 60) * 1000  # before 09:30 ET in ms
        ]
        if not pm_bars:
            return None

        return {
            "pre_market_high": max(b["h"] for b in pm_bars),
            "pre_market_low": min(b["l"] for b in pm_bars),
            "pre_market_volume": sum(b["v"] for b in pm_bars),
        }
    except Exception as exc:
        log.debug("premarket_failed", ticker=ticker, date=target_date, error=str(exc))
        return None


def fetch_grouped_daily(target_date: str) -> list[dict]:
    """
    Polygon /v2/aggs/grouped/locale/us/market/stocks — returns all tickers
    in one call for the given date.  Much faster than per-ticker calls for
    bulk ingestion.
    """
    try:
        data = _get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{target_date}",
            {"adjusted": "true", "include_otc": "false"},
        )
        return data.get("results", [])
    except Exception as exc:
        log.error("grouped_daily_failed", date=target_date, error=str(exc))
        return []


# ── Derived metrics ────────────────────────────────────────────────────────────

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of daily bars with columns
    [date, ticker, open, high, low, close, volume, vwap],
    add: prior_close, gap_pct, intraday_range_pct, avg_volume_20d, relative_volume.

    Assumes df is sorted by (ticker, date).
    """
    df = df.sort_values(["ticker", "date"]).copy()

    df["prior_close"] = df.groupby("ticker")["close"].shift(1)
    df["gap_pct"] = (df["open"] - df["prior_close"]) / df["prior_close"].replace(0, float("nan"))
    df["intraday_range_pct"] = (df["high"] - df["low"]) / df["open"].replace(0, float("nan"))

    # Rolling 20-day average volume (using historical window ending *before* current bar)
    df["avg_volume_20d"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
    df["relative_volume"] = df["volume"] / df["avg_volume_20d"].replace(0, float("nan"))

    return df


# ── Main ingestion functions ───────────────────────────────────────────────────

def ingest_grouped_daily(target_date: str, db_path=None) -> int:
    """
    Fast path: use Polygon's grouped daily endpoint to load all stocks
    in a single API call, then filter to universe tickers.

    Returns the number of rows written.
    """
    log.info("ingest_grouped_daily_start", date=target_date)

    # Load the universe for this date
    conn = get_connection(db_path)
    universe_df = fetch_df(
        conn,
        "SELECT ticker FROM universe WHERE date = ? AND is_active = 1",
        (target_date,),
    )
    if universe_df.empty:
        log.warning("no_universe_for_date", date=target_date)
        return 0

    universe_tickers: set[str] = set(universe_df["ticker"].tolist())

    # Fetch grouped bars
    bars = fetch_grouped_daily(target_date)
    if not bars:
        log.warning("no_grouped_bars", date=target_date)
        return 0

    rows = []
    for b in bars:
        ticker = b.get("T", "")
        if ticker not in universe_tickers:
            continue
        rows.append({
            "date": target_date,
            "ticker": ticker,
            "open": b.get("o"),
            "high": b.get("h"),
            "low": b.get("l"),
            "close": b.get("c"),
            "volume": b.get("v"),
            "vwap": b.get("vw"),
            "avg_volume_20d": None,     # computed in post-process step
            "relative_volume": None,
            "gap_pct": None,
            "intraday_range_pct": None,
            "prior_close": None,
            "pre_market_high": None,
            "pre_market_low": None,
            "pre_market_volume": None,
        })

    with get_db(db_path) as conn:
        inserted = upsert_rows(conn, "daily_prices", rows)

    log.info("ingest_grouped_daily_done", date=target_date, rows=len(rows))
    return len(rows)


def backfill_ticker(
    ticker: str,
    from_date: str,
    to_date: str,
    db_path=None,
) -> int:
    """
    Per-ticker backfill: fetch all daily bars for *ticker* over a date range,
    compute derived metrics, and write to daily_prices.
    """
    bars = fetch_daily_bars(ticker, from_date, to_date)
    if not bars:
        return 0

    records = []
    for b in bars:
        # Polygon bar timestamp is milliseconds since epoch
        ts_ms = b.get("t", 0)
        bar_date = date.fromtimestamp(ts_ms / 1000).isoformat()
        records.append({
            "date": bar_date,
            "ticker": ticker,
            "open": b.get("o"),
            "high": b.get("h"),
            "low": b.get("l"),
            "close": b.get("c"),
            "volume": b.get("v"),
            "vwap": b.get("vw"),
            "avg_volume_20d": None,
            "relative_volume": None,
            "gap_pct": None,
            "intraday_range_pct": None,
            "prior_close": None,
            "pre_market_high": None,
            "pre_market_low": None,
            "pre_market_volume": None,
        })

    df = pd.DataFrame(records)
    df = compute_derived(df)

    rows = df.to_dict("records")
    with get_db(db_path) as conn:
        upsert_rows(conn, "daily_prices", rows)

    return len(rows)


def compute_and_store_derived(db_path=None) -> None:
    """
    Post-process step: read all rows in daily_prices where derived fields
    are NULL, compute them in bulk, and write back.

    Run once after a grouped-daily bulk load.
    """
    log.info("computing_derived_fields")
    conn = get_connection(db_path)

    df = fetch_df(conn, "SELECT * FROM daily_prices ORDER BY ticker, date")
    if df.empty:
        return

    df = compute_derived(df)

    # Only update rows where we now have values
    # Replace NaN with None for SQLite
    df = df.where(pd.notnull(df), None)

    rows = df.to_dict("records")
    with get_db(db_path) as conn:
        upsert_rows(conn, "daily_prices", rows)

    log.info("derived_fields_done", rows=len(rows))


def ingest_premarket(tickers: list[str], target_date: str, db_path=None) -> int:
    """
    Fetch and store pre-market data for a list of tickers on *target_date*.
    Typically called for the scanner watchlist (high-priority names only).
    """
    updated = 0
    for ticker in tickers:
        pm = fetch_premarket_bars(ticker, target_date)
        if pm is None:
            continue
        with get_db(db_path) as conn:
            conn.execute(
                """
                UPDATE daily_prices
                   SET pre_market_high   = ?,
                       pre_market_low    = ?,
                       pre_market_volume = ?
                 WHERE date = ? AND ticker = ?
                """,
                (
                    pm["pre_market_high"],
                    pm["pre_market_low"],
                    pm["pre_market_volume"],
                    target_date,
                    ticker,
                ),
            )
        updated += 1
    log.info("premarket_updated", date=target_date, count=updated)
    return updated


def ingest_date_range(
    from_date: str,
    to_date: str | None = None,
    db_path=None,
) -> None:
    """
    Full bulk ingestion: for each trading day in [from_date, to_date],
    call ingest_grouped_daily(), then run compute_and_store_derived().
    """
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date) if to_date else date.today()

    current = start
    while current <= end:
        if current.weekday() < 5:
            ds = current.isoformat()
            try:
                ingest_grouped_daily(ds, db_path)
            except Exception as exc:
                log.error("ingest_date_failed", date=ds, error=str(exc))
        current += timedelta(days=1)

    compute_and_store_derived(db_path)
