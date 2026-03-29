"""
src/pipeline/universe.py

Builds the point-in-time stock universe for each trading day.

Sources
-------
- Polygon.io  /v3/reference/tickers  (active + delisted symbols)
- Polygon.io  /v2/aggs/ticker/{ticker}/range  (for avg volume)

The output is written to the ``universe`` table in SQLite.

Survivorship-bias note
----------------------
We fetch both active and delisted tickers from Polygon so that
the backtest never has access to a "survivor-only" universe.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import blocked_tickers, settings
from src.utils.db import get_db, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)

_POLYGON_BASE = settings["api"]["polygon"]["base_url"]
_API_KEY = settings["api"]["polygon"]["api_key"]
_RATE_LIMIT = settings["api"]["polygon"]["rate_limit_per_minute"]

# Seconds between requests to respect rate limit
_REQUEST_GAP = 60.0 / max(_RATE_LIMIT, 1)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(endpoint: str, params: dict | None = None) -> dict:
    params = params or {}
    params["apiKey"] = _API_KEY
    url = f"{_POLYGON_BASE}{endpoint}"
    resp = requests.get(url, params=params, timeout=settings["pipeline"]["request_timeout"])
    resp.raise_for_status()
    time.sleep(_REQUEST_GAP)
    return resp.json()


def _paginate(endpoint: str, params: dict | None = None) -> list[dict]:
    """Follow Polygon's cursor-based pagination and return all results."""
    params = params or {}
    results: list[dict] = []
    while True:
        data = _get(endpoint, params)
        results.extend(data.get("results", []))
        next_url = data.get("next_url")
        if not next_url:
            break
        # next_url includes the cursor; parse it out
        cursor = next_url.split("cursor=")[-1].split("&")[0]
        params = {"cursor": cursor}
    return results


# ── Universe construction ──────────────────────────────────────────────────────

def fetch_ticker_details(ticker: str) -> dict | None:
    """Return Polygon ticker details (market cap, exchange, name)."""
    try:
        data = _get(f"/v3/reference/tickers/{ticker}")
        return data.get("results")
    except Exception as exc:
        log.warning("ticker_details_failed", ticker=ticker, error=str(exc))
        return None


def fetch_all_tickers(
    market: str = "stocks",
    exchange: str | None = None,
    active: bool | None = None,
) -> list[dict]:
    """
    Fetch all tickers from Polygon reference endpoint.

    active=None → fetches both active and delisted (for survivorship bias).
    """
    params: dict[str, Any] = {
        "market": market,
        "limit": 1000,
    }
    if exchange:
        params["exchange"] = exchange
    if active is not None:
        params["active"] = str(active).lower()

    log.info("fetching_tickers", market=market, exchange=exchange, active=active)
    return _paginate("/v3/reference/tickers", params)


def compute_avg_volume(ticker: str, as_of: date, lookback: int = 20) -> float | None:
    """
    Compute 20-day average daily volume for *ticker* ending on *as_of*.
    Returns None if insufficient data.
    """
    from_date = (as_of - timedelta(days=lookback * 2)).isoformat()  # extra buffer for holidays
    to_date = as_of.isoformat()
    try:
        data = _get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
            {"adjusted": "true", "sort": "desc", "limit": lookback},
        )
        results = data.get("results", [])
        if not results:
            return None
        volumes = [r["v"] for r in results if "v" in r]
        if len(volumes) < 5:
            return None
        return sum(volumes) / len(volumes)
    except Exception as exc:
        log.debug("avg_volume_failed", ticker=ticker, error=str(exc))
        return None


def _passes_filters(
    ticker: str,
    details: dict,
    avg_vol: float | None,
) -> bool:
    """Apply universe filter criteria to a single ticker."""
    u = settings["universe"]

    if ticker in blocked_tickers:
        return False

    # Exchange
    exchange = (details.get("primary_exchange") or "").upper()
    allowed_exchanges = {e.upper() for e in u["exchanges"]}
    # Polygon uses MIC codes like XNYS (NYSE) and XNAS (NASDAQ)
    exchange_map = {"XNYS": "NYSE", "XNAS": "NASDAQ", "NYSE": "NYSE", "NASDAQ": "NASDAQ"}
    normalized = exchange_map.get(exchange, exchange)
    if normalized not in allowed_exchanges:
        return False

    # Market cap
    market_cap = details.get("market_cap") or 0
    if not (u["market_cap_min"] <= market_cap <= u["market_cap_max"]):
        return False

    # Price filter is applied later in price_volume.py (we don't have price here)

    # Volume
    if avg_vol is not None and avg_vol < u["min_avg_volume"]:
        return False

    return True


def build_universe_for_date(target_date: date, db_path=None) -> int:
    """
    Build the universe for a single trading date and write to the DB.
    Returns the number of tickers inserted.
    """
    log.info("building_universe", date=target_date.isoformat())

    # 1. Fetch all tickers (active + inactive for survivorship-bias avoidance)
    all_tickers = fetch_all_tickers(market="stocks")
    if settings["universe"]["include_delisted"]:
        all_tickers += fetch_all_tickers(market="stocks", active=False)

    # Deduplicate
    seen: set[str] = set()
    unique_tickers = []
    for t in all_tickers:
        sym = t.get("ticker", "")
        if sym and sym not in seen:
            seen.add(sym)
            unique_tickers.append(t)

    log.info("tickers_fetched", count=len(unique_tickers))

    rows: list[dict] = []
    for t in unique_tickers:
        ticker = t.get("ticker", "")
        if not ticker:
            continue

        # Compute avg volume
        avg_vol = compute_avg_volume(ticker, target_date)

        if not _passes_filters(ticker, t, avg_vol):
            continue

        exchange = (t.get("primary_exchange") or "").upper()
        exchange_map = {"XNYS": "NYSE", "XNAS": "NASDAQ"}
        normalized_exchange = exchange_map.get(exchange, exchange)

        rows.append({
            "date": target_date.isoformat(),
            "ticker": ticker,
            "company_name": t.get("name"),
            "market_cap": t.get("market_cap"),
            "sector": t.get("sic_description"),
            "exchange": normalized_exchange,
            "avg_volume_20d": avg_vol,
            "is_active": 1 if t.get("active", True) else 0,
        })

    with get_db(db_path) as conn:
        inserted = upsert_rows(conn, "universe", rows)

    log.info("universe_built", date=target_date.isoformat(), tickers=len(rows))
    return len(rows)


def build_universe_range(
    start: str | date,
    end: str | date | None = None,
    db_path=None,
) -> None:
    """
    Build the universe for every trading day in [start, end].
    Skips weekends; does not yet skip market holidays (TODO: integrate trading calendar).
    """
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if end is None:
        end = date.today()
    elif isinstance(end, str):
        end = date.fromisoformat(end)

    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon–Fri
            try:
                build_universe_for_date(current, db_path)
            except Exception as exc:
                log.error("universe_date_failed", date=current.isoformat(), error=str(exc))
        current += timedelta(days=1)
